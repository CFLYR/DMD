#!/bin/bash
# DMD Batch Experiments Runner
# One-click script to run all experiments or specific stages

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DMD_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DMD Experiment Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        return 0
    else
        print_error "Python 3 not found!"
        return 1
    fi
}

# Check GPU availability
check_gpu() {
    print_info "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/,$//')
        print_success "Found $GPU_COUNT GPU(s): $GPU_NAMES"
        return 0
    else
        print_warning "nvidia-smi not found. GPU may not be available."
        return 1
    fi
}

# Check dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    cd "$DMD_ROOT"
    
    REQUIRED_PACKAGES=("torch" "numpy" "pandas" "easydict")
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
        print_success "All required packages are installed"
        return 0
    else
        print_error "Missing packages: ${MISSING_PACKAGES[*]}"
        print_info "Install with: pip install -r requirements.txt"
        return 1
    fi
}

# Generate configurations
generate_configs() {
    print_info "Generating experiment configurations..."
    cd "$DMD_ROOT"
    python3 scripts/config_generator.py
    if [ $? -eq 0 ]; then
        print_success "Configurations generated successfully"
        return 0
    else
        print_error "Configuration generation failed"
        return 1
    fi
}

# Run smoke test
run_smoke_test() {
    local epochs=${1:-2}
    print_info "Running smoke test ($epochs epochs)..."
    cd "$DMD_ROOT"
    python3 scripts/smoke_test.py --epochs "$epochs"
    return $?
}

# Run batch training
run_batch_train() {
    print_info "Starting batch training..."
    print_warning "This will take several hours. Please ensure GPU is available."
    cd "$DMD_ROOT"
    python3 scripts/batch_train.py
    return $?
}

# Run batch testing
run_batch_test() {
    print_info "Running batch testing..."
    cd "$DMD_ROOT"
    python3 scripts/batch_test.py
    return $?
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --smoke [epochs]    Run smoke test (default: 2 epochs)
    --train            Run full batch training
    --test             Run batch testing
    --all              Run complete pipeline (configs + train + test)
    --check            Check environment only
    --help             Show this help message

Examples:
    $0 --check                  # Check environment
    $0 --smoke 3                # Quick test with 3 epochs
    $0 --train                  # Run full training
    $0 --test                   # Test trained models
    $0 --all                    # Complete pipeline

EOF
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

MODE=""
EPOCHS=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            MODE="smoke"
            if [[ $2 =~ ^[0-9]+$ ]]; then
                EPOCHS=$2
                shift
            fi
            shift
            ;;
        --train)
            MODE="train"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --check)
            MODE="check"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
echo ""
print_info "Starting environment checks..."
echo ""

# Always check environment
check_python || exit 1
check_gpu
check_dependencies || exit 1

if [ "$MODE" = "check" ]; then
    echo ""
    print_success "Environment check complete!"
    exit 0
fi

echo ""
print_info "Mode: $MODE"
echo ""

case $MODE in
    smoke)
        generate_configs || exit 1
        echo ""
        run_smoke_test "$EPOCHS"
        EXIT_CODE=$?
        echo ""
        if [ $EXIT_CODE -eq 0 ]; then
            print_success "Smoke test passed!"
        else
            print_error "Smoke test failed!"
        fi
        exit $EXIT_CODE
        ;;
    
    train)
        generate_configs || exit 1
        echo ""
        read -p "This will start full training (several hours). Continue? (y/N) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_batch_train
            EXIT_CODE=$?
            echo ""
            if [ $EXIT_CODE -eq 0 ]; then
                print_success "Batch training complete!"
            else
                print_error "Batch training failed!"
            fi
            exit $EXIT_CODE
        else
            print_info "Training cancelled"
            exit 0
        fi
        ;;
    
    test)
        run_batch_test
        EXIT_CODE=$?
        echo ""
        if [ $EXIT_CODE -eq 0 ]; then
            print_success "Batch testing complete!"
        else
            print_error "Batch testing failed!"
        fi
        exit $EXIT_CODE
        ;;
    
    all)
        generate_configs || exit 1
        echo ""
        print_info "Running complete pipeline..."
        print_warning "This will take several hours!"
        echo ""
        read -p "Continue? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Pipeline cancelled"
            exit 0
        fi
        
        # Run smoke test first
        print_info "Step 1/3: Smoke test..."
        run_smoke_test 2 || {
            print_error "Smoke test failed! Aborting pipeline."
            exit 1
        }
        
        echo ""
        print_info "Step 2/3: Batch training..."
        run_batch_train || {
            print_error "Batch training failed! Aborting pipeline."
            exit 1
        }
        
        echo ""
        print_info "Step 3/3: Batch testing..."
        run_batch_test || {
            print_error "Batch testing failed!"
            exit 1
        }
        
        echo ""
        print_success "Complete pipeline finished successfully!"
        exit 0
        ;;
    
    *)
        print_error "Invalid mode: $MODE"
        usage
        exit 1
        ;;
esac
