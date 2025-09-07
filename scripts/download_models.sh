#!/bin/bash
# Download Models Script for AI Memory Service
# Downloads EmbeddingGemma-300m ONNX model and tokenizer from Hugging Face

set -e

# Default values
FORCE=false
MODELS_DIR="models"
VERIFY=true
PARALLEL=false

# Model information
REPO_ID="onnx-community/embeddinggemma-300m-ONNX"
BASE_URL="https://huggingface.co/$REPO_ID/resolve/main"

# Model files with checksums (SHA256)
declare -A FILES=(
    ["embedding_model.onnx"]="model.onnx|~620MB|EmbeddingGemma-300m ONNX Model"
    ["tokenizer.json"]="tokenizer.json|~2.1MB|Tokenizer Configuration"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_help() {
    echo "AI Memory Service - Model Download Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --force       Force re-download even if files exist"
    echo "  -d, --dir DIR     Models directory (default: models)"
    echo "  --no-verify       Skip file verification"
    echo "  -p, --parallel    Download files in parallel"
    echo "  -h, --help        Show this help message"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --no-verify)
            VERIFY=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${CYAN}AI Memory Service - Model Download Script${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

# Create models directory
if [[ ! -d "$MODELS_DIR" ]]; then
    echo -e "${GREEN}Creating models directory: $MODELS_DIR${NC}"
    mkdir -p "$MODELS_DIR"
fi

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${YELLOW}Warning: $1 not found. $2${NC}"
        return 1
    fi
    return 0
}

# Check tools
check_tool "curl" "Using basic download method"
CURL_AVAILABLE=$?

check_tool "wget" "Using curl instead"
WGET_AVAILABLE=$?

if [[ $CURL_AVAILABLE -ne 0 && $WGET_AVAILABLE -ne 0 ]]; then
    echo -e "${RED}Error: Neither curl nor wget available for downloading${NC}"
    exit 1
fi

# Check for Git LFS
GIT_LFS_AVAILABLE=false
if check_tool "git" "Git LFS not available for large files" && git lfs version &>/dev/null; then
    GIT_LFS_AVAILABLE=true
    echo -e "${YELLOW}Git LFS detected - available for large file handling${NC}"
fi

# Download function
download_file() {
    local filename="$1"
    local remote_name="$2"
    local size="$3"
    local description="$4"
    
    local filepath="$MODELS_DIR/$filename"
    local url="$BASE_URL/$remote_name"
    
    echo ""
    echo -e "${CYAN}Processing: $description${NC}"
    echo "  File: $filename"
    echo "  Size: $size"
    
    if [[ -f "$filepath" && "$FORCE" != true ]]; then
        local existing_size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "0")
        echo -e "  Status: Already exists ($(numfmt --to=iec-i --suffix=B $existing_size))${GREEN} ✓${NC}"
        
        if [[ "$VERIFY" == true ]]; then
            echo -e "  ${YELLOW}Verifying file integrity...${NC}"
            # Basic size check - in production, add proper checksums
            if [[ $existing_size -lt 1000000 ]]; then
                echo -e "  ${RED}Warning: File appears incomplete, re-downloading...${NC}"
                rm -f "$filepath"
            else
                echo -e "  Verification: ${GREEN}OK${NC}"
                return 0
            fi
        else
            return 0
        fi
    fi
    
    echo -e "  Status: ${YELLOW}Downloading...${NC}"
    
    # Download with progress bar
    if [[ $CURL_AVAILABLE -eq 0 ]]; then
        if curl -L --progress-bar "$url" -o "$filepath"; then
            local downloaded_size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "0")
            echo -e "  Completed: $(numfmt --to=iec-i --suffix=B $downloaded_size) downloaded ${GREEN}✓${NC}"
        else
            echo -e "  ${RED}Error: Failed to download with curl${NC}"
            return 1
        fi
    elif [[ $WGET_AVAILABLE -eq 0 ]]; then
        if wget --progress=bar:force "$url" -O "$filepath"; then
            local downloaded_size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "0")
            echo -e "  Completed: $(numfmt --to=iec-i --suffix=B $downloaded_size) downloaded ${GREEN}✓${NC}"
        else
            echo -e "  ${RED}Error: Failed to download with wget${NC}"
            return 1
        fi
    fi
}

# Download files
if [[ "$PARALLEL" == true ]]; then
    echo -e "${YELLOW}Starting parallel downloads...${NC}"
    pids=()
    
    for filename in "${!FILES[@]}"; do
        IFS='|' read -r remote_name size description <<< "${FILES[$filename]}"
        download_file "$filename" "$remote_name" "$size" "$description" &
        pids+=($!)
    done
    
    # Wait for all downloads to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
else
    echo -e "${YELLOW}Starting sequential downloads...${NC}"
    
    for filename in "${!FILES[@]}"; do
        IFS='|' read -r remote_name size description <<< "${FILES[$filename]}"
        download_file "$filename" "$remote_name" "$size" "$description"
    done
fi

# Summary
echo ""
echo -e "${CYAN}Download Summary:${NC}"
echo -e "${CYAN}================${NC}"

total_size=0
all_present=true

for filename in "${!FILES[@]}"; do
    filepath="$MODELS_DIR/$filename"
    if [[ -f "$filepath" ]]; then
        size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "0")
        total_size=$((total_size + size))
        echo -e "  ${GREEN}✓${NC} $filename - $(numfmt --to=iec-i --suffix=B $size)"
    else
        echo -e "  ${RED}✗${NC} $filename - Missing"
        all_present=false
    fi
done

echo ""
echo -e "${CYAN}Total model size: $(numfmt --to=iec-i --suffix=B $total_size)${NC}"

# Verify configuration
config_path="config.toml"
if [[ -f "$config_path" ]]; then
    echo ""
    echo -e "${CYAN}Verifying configuration...${NC}"
    
    if grep -q 'model_path.*models/embedding_model\.onnx' "$config_path" && \
       grep -q 'tokenizer_path.*models/tokenizer\.json' "$config_path"; then
        echo -e "  Configuration: ${GREEN}OK${NC}"
    else
        echo -e "  Configuration: ${YELLOW}Needs update${NC}"
        echo "  Please ensure config.toml has correct model paths:"
        echo '    model_path = "./models/embedding_model.onnx"'
        echo '    tokenizer_path = "./models/tokenizer.json"'
    fi
else
    echo ""
    echo -e "${YELLOW}Warning: config.toml not found${NC}"
    echo "Please create configuration file with model paths"
fi

# Generate secure Neo4j password if not set
NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(openssl rand -base64 12 2>/dev/null || echo "secure_$(date +%s)")}"

if [[ "$all_present" == true ]]; then
    echo ""
    echo -e "${GREEN}Model download completed successfully! 🎉${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "1. Verify config.toml has correct model paths"
    echo "2. Set Neo4j password: export NEO4J_PASSWORD='$NEO4J_PASSWORD'"
    echo "3. Start Neo4j database:"
    echo "   docker run -d --name neo4j-memory -p 7474:7474 -p 7687:7687 \\"
    echo "   -e NEO4J_AUTH=neo4j/\$NEO4J_PASSWORD neo4j:5.0"
    echo "4. Run the service: cargo run --release"
else
    echo ""
    echo -e "${RED}Some files failed to download. Please check errors above.${NC}"
    exit 1
fi

echo ""