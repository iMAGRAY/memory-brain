//! Graceful shutdown system for AI Memory Service
//!
//! Provides coordinated shutdown handling for Windows and Unix systems
//! with proper cleanup of resources, connections, and background tasks.

use std::sync::{Arc, OnceLock, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::{timeout, sleep};
use tracing::{info, warn, error, debug};
use crate::types::MemoryResult;

/// Shutdown signal types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShutdownReason {
    /// SIGTERM or Ctrl+C
    Interrupt,
    /// SIGQUIT
    Quit,
    /// Application error
    Error,
    /// Manual shutdown request
    Manual,
    /// System shutdown
    System,
}

/// Shutdown hook priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShutdownPriority {
    /// Critical infrastructure (metrics, logging)
    Critical = 0,
    /// Core services (API, database connections)
    High = 1,
    /// Business logic (memory service, embeddings)
    Normal = 2,
    /// Background tasks (cleanup, monitoring)
    Low = 3,
    /// Optional components (caches, temp files)
    Background = 4,
}

/// Shutdown hook function signature
pub type ShutdownHook = Box<dyn Fn(ShutdownReason) -> futures::future::BoxFuture<'static, MemoryResult<()>> + Send + Sync>;

/// Component to be shut down
struct ShutdownComponent {
    name: String,
    priority: ShutdownPriority,
    timeout: Duration,
    hook: ShutdownHook,
    completed: Arc<AtomicBool>,
}

/// Graceful shutdown coordinator
pub struct ShutdownCoordinator {
    /// Shutdown signal broadcaster
    shutdown_tx: broadcast::Sender<ShutdownReason>,
    
    /// Registered shutdown components
    components: Arc<RwLock<Vec<Arc<ShutdownComponent>>>>,
    
    /// Shutdown state
    is_shutting_down: Arc<AtomicBool>,
    shutdown_start_time: Arc<Mutex<Option<Instant>>>,
    
    /// Configuration
    global_timeout: Duration,
    #[allow(dead_code)]
    force_kill_timeout: Duration,
}

impl ShutdownCoordinator {
    /// Create new shutdown coordinator
    pub fn new() -> Self {
        let (shutdown_tx, _) = broadcast::channel(16);
        
        ShutdownCoordinator {
            shutdown_tx,
            components: Arc::new(RwLock::new(Vec::new())),
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            shutdown_start_time: Arc::new(Mutex::new(None)),
            global_timeout: Duration::from_secs(30),
            force_kill_timeout: Duration::from_secs(60),
        }
    }
    
    /// Register a component for graceful shutdown
    pub async fn register_component<F>(&self, name: String, priority: ShutdownPriority, timeout: Duration, hook: F)
    where
        F: Fn(ShutdownReason) -> futures::future::BoxFuture<'static, MemoryResult<()>> + Send + Sync + 'static,
    {
        let component = Arc::new(ShutdownComponent {
            name: name.clone(),
            priority,
            timeout,
            hook: Box::new(hook),
            completed: Arc::new(AtomicBool::new(false)),
        });
        
        let mut components = self.components.write().await;
        components.push(component);
        
        // Sort by priority for ordered shutdown
        components.sort_by_key(|c| c.priority);
        
        info!("Registered shutdown component: {} (priority: {:?})", name, priority);
    }
    
    /// Get shutdown signal receiver
    pub fn subscribe(&self) -> broadcast::Receiver<ShutdownReason> {
        self.shutdown_tx.subscribe()
    }
    
    /// Check if system is shutting down
    pub fn is_shutting_down(&self) -> bool {
        self.is_shutting_down.load(Ordering::Acquire)
    }
    
    /// Initiate graceful shutdown
    pub async fn shutdown(&self, reason: ShutdownReason) -> MemoryResult<()> {
        // Prevent multiple shutdowns
        if self.is_shutting_down.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            warn!("Shutdown already in progress");
            return Ok(());
        }
        
        // Record shutdown start time
        *self.shutdown_start_time.lock().await = Some(Instant::now());
        
        info!("Initiating graceful shutdown (reason: {:?})", reason);
        
        // Broadcast shutdown signal
        if let Err(e) = self.shutdown_tx.send(reason) {
            warn!("Failed to broadcast shutdown signal: {}", e);
        }
        
        // Execute shutdown hooks in priority order
        let components = self.components.read().await;
        let mut shutdown_tasks = Vec::new();
        
        for component in components.iter() {
            let comp = Arc::clone(component);
            let task = tokio::spawn(async move {
                Self::execute_component_shutdown(comp, reason).await
            });
            shutdown_tasks.push(task);
        }
        
        // Wait for all components to shut down with global timeout
        match timeout(self.global_timeout, futures::future::join_all(shutdown_tasks)).await {
            Ok(results) => {
                let mut failed_components = Vec::new();
                for (i, result) in results.into_iter().enumerate() {
                    match result {
                        Ok(Ok(())) => {},
                        Ok(Err(e)) => {
                            failed_components.push((components[i].name.clone(), e.to_string()));
                        },
                        Err(e) => {
                            failed_components.push((components[i].name.clone(), format!("Task error: {}", e)));
                        }
                    }
                }
                
                if !failed_components.is_empty() {
                    warn!("Some components failed to shutdown gracefully: {:?}", failed_components);
                } else {
                    info!("All components shut down successfully");
                }
            },
            Err(_) => {
                error!("Global shutdown timeout reached, some components may not have shut down properly");
            }
        }
        
        if let Some(start_time) = *self.shutdown_start_time.lock().await {
            let shutdown_duration = start_time.elapsed();
            info!("Shutdown completed in {:?}", shutdown_duration);
        }
        
        Ok(())
    }
    
    /// Execute shutdown for a single component
    async fn execute_component_shutdown(component: Arc<ShutdownComponent>, reason: ShutdownReason) -> MemoryResult<()> {
        debug!("Shutting down component: {}", component.name);
        let start_time = Instant::now();
        
        // Execute the shutdown hook with timeout
        let hook_future = (component.hook)(reason);
        
        match timeout(component.timeout, hook_future).await {
            Ok(result) => {
                match result {
                    Ok(()) => {
                        component.completed.store(true, Ordering::Release);
                        let duration = start_time.elapsed();
                        info!("Component '{}' shut down successfully in {:?}", component.name, duration);
                        Ok(())
                    },
                    Err(e) => {
                        error!("Component '{}' shutdown failed: {}", component.name, e);
                        Err(e)
                    }
                }
            },
            Err(_) => {
                error!("Component '{}' shutdown timed out after {:?}", component.name, component.timeout);
                Err(format!("Shutdown timeout for component: {}", component.name).into())
            }
        }
    }
    
    /// Force immediate shutdown (last resort)
    pub async fn force_shutdown(&self) -> ! {
        warn!("Force shutdown initiated - terminating immediately");
        
        // Give a brief moment for logging to flush
        sleep(Duration::from_millis(100)).await;
        
        std::process::exit(1);
    }
    
    /// Get shutdown statistics
    pub async fn get_shutdown_stats(&self) -> ShutdownStats {
        let components = self.components.read().await;
        let start_time = *self.shutdown_start_time.lock().await;
        
        let mut stats = ShutdownStats {
            total_components: components.len(),
            completed_components: 0,
            failed_components: 0,
            is_shutting_down: self.is_shutting_down(),
            shutdown_duration: None,
            components_status: Vec::new(),
        };
        
        for component in components.iter() {
            let completed = component.completed.load(Ordering::Acquire);
            if completed {
                stats.completed_components += 1;
            } else if self.is_shutting_down() {
                stats.failed_components += 1;
            }
            
            stats.components_status.push(ComponentStatus {
                name: component.name.clone(),
                priority: component.priority,
                completed,
            });
        }
        
        if let Some(start) = start_time {
            stats.shutdown_duration = Some(start.elapsed());
        }
        
        stats
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self { Self::new() }
}

/// Shutdown statistics
#[derive(Debug, Clone)]
pub struct ShutdownStats {
    pub total_components: usize,
    pub completed_components: usize,
    pub failed_components: usize,
    pub is_shutting_down: bool,
    pub shutdown_duration: Option<Duration>,
    pub components_status: Vec<ComponentStatus>,
}

/// Component shutdown status
#[derive(Debug, Clone)]
pub struct ComponentStatus {
    pub name: String,
    pub priority: ShutdownPriority,
    pub completed: bool,
}

/// Global shutdown coordinator instance
static SHUTDOWN_COORDINATOR: OnceLock<Arc<ShutdownCoordinator>> = OnceLock::new();

impl ShutdownCoordinator {
    /// Initialize global shutdown coordinator
    pub fn initialize() -> Arc<Self> {
        SHUTDOWN_COORDINATOR.get_or_init(|| {
            Arc::new(ShutdownCoordinator::new())
        }).clone()
    }
    
    /// Get global shutdown coordinator
    pub fn global() -> Option<Arc<Self>> {
        SHUTDOWN_COORDINATOR.get().cloned()
    }
}

/// Signal handler setup for different platforms
pub async fn setup_signal_handlers(coordinator: Arc<ShutdownCoordinator>) -> MemoryResult<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        
        // SIGTERM handler
        let mut sigterm = signal(SignalKind::terminate())?;
        let coord_term = Arc::clone(&coordinator);
        tokio::spawn(async move {
            sigterm.recv().await;
            info!("Received SIGTERM");
            if let Err(e) = coord_term.shutdown(ShutdownReason::Interrupt).await {
                error!("Failed to handle SIGTERM: {}", e);
            }
        });
        
        // SIGQUIT handler
        let mut sigquit = signal(SignalKind::quit())?;
        let coord_quit = Arc::clone(&coordinator);
        tokio::spawn(async move {
            sigquit.recv().await;
            info!("Received SIGQUIT");
            if let Err(e) = coord_quit.shutdown(ShutdownReason::Quit).await {
                error!("Failed to handle SIGQUIT: {}", e);
            }
        });
    }
    
    // Ctrl+C handler (works on both Windows and Unix)
    let coord_int = Arc::clone(&coordinator);
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!("Failed to listen for Ctrl+C: {}", e);
            return;
        }
        info!("Received Ctrl+C");
        if let Err(e) = coord_int.shutdown(ShutdownReason::Interrupt).await {
            error!("Failed to handle Ctrl+C: {}", e);
        }
    });
    
    #[cfg(windows)]
    {
        // Windows-specific signal handling could be added here
        // For now, Ctrl+C handler covers the main case
        info!("Signal handlers setup for Windows (Ctrl+C)");
    }
    
    #[cfg(unix)]
    {
        info!("Signal handlers setup for Unix (SIGTERM, SIGQUIT, SIGINT)");
    }
    
    Ok(())
}

/// High-level shutdown management API
pub struct ShutdownManager;

impl ShutdownManager {
    /// Initialize shutdown system
    pub async fn init() -> MemoryResult<Arc<ShutdownCoordinator>> {
        let coordinator = ShutdownCoordinator::initialize();
        setup_signal_handlers(Arc::clone(&coordinator)).await?;
        info!("Shutdown manager initialized");
        Ok(coordinator)
    }
    
    /// Register component for graceful shutdown
    pub async fn register<F>(name: &str, priority: ShutdownPriority, hook: F) -> MemoryResult<()>
    where
        F: Fn(ShutdownReason) -> std::pin::Pin<Box<dyn std::future::Future<Output = MemoryResult<()>> + Send>> + Send + Sync + 'static,
    {
        if let Some(coordinator) = ShutdownCoordinator::global() {
            coordinator.register_component(
                name.to_string(),
                priority,
                Duration::from_secs(10), // Default component timeout
                hook,
            ).await;
            Ok(())
        } else {
            Err("Shutdown coordinator not initialized".into())
        }
    }
    
    /// Check if system is shutting down
    pub fn is_shutting_down() -> bool {
        ShutdownCoordinator::global()
            .map(|c| c.is_shutting_down())
            .unwrap_or(false)
    }
    
    /// Manual shutdown request
    pub async fn shutdown() -> MemoryResult<()> {
        if let Some(coordinator) = ShutdownCoordinator::global() {
            coordinator.shutdown(ShutdownReason::Manual).await
        } else {
            Err("Shutdown coordinator not initialized".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    
    #[tokio::test]
    async fn test_shutdown_coordinator() {
        let coordinator = ShutdownCoordinator::new();
        
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);
        
        coordinator.register_component(
            "test_component".to_string(),
            ShutdownPriority::Normal,
            Duration::from_secs(1),
            move |_reason| {
                let counter = Arc::clone(&counter_clone);
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })
            },
        ).await;
        
        assert!(!coordinator.is_shutting_down());
        
        coordinator.shutdown(ShutdownReason::Manual).await.unwrap();
        
        assert!(coordinator.is_shutting_down());
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
    
    #[tokio::test]
    async fn test_shutdown_priority_order() {
        let coordinator = ShutdownCoordinator::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        
        // Register components in reverse priority order
        for (name, priority) in [
            ("background", ShutdownPriority::Background),
            ("critical", ShutdownPriority::Critical),
            ("normal", ShutdownPriority::Normal),
            ("high", ShutdownPriority::High),
        ] {
            let order_clone = Arc::clone(&order);
            let name_owned = name.to_string();
            coordinator.register_component(
                name_owned.clone(),
                priority,
                Duration::from_secs(1),
                move |_reason| {
                    let order = Arc::clone(&order_clone);
                    let name = name_owned.clone();
                    Box::pin(async move {
                        order.lock().await.push(name);
                        Ok(())
                    })
                },
            ).await;
        }
        
        coordinator.shutdown(ShutdownReason::Manual).await.unwrap();
        
        let final_order = order.lock().await;
        // Should be in priority order: Critical, High, Normal, Background
        assert_eq!(final_order[0], "critical");
        assert_eq!(final_order[1], "high");
        assert_eq!(final_order[2], "normal");
        assert_eq!(final_order[3], "background");
    }
}
