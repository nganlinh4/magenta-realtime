#!/bin/bash

# Magenta RT Setup Script
# This script automatically sets up the real Magenta RT system

set -e  # Exit on any error

echo "🎵 Magenta RT Setup Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "backend" ]; then
    print_error "Please run this script from the magenta-realtime project root directory"
    exit 1
fi

print_status "Checking Magenta RT requirements..."

# Check 1: Virtual environment
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run 'npm run setup' first."
    exit 1
fi

print_success "Virtual environment found"

# Check 2: Magenta RT package installation
source venv/bin/activate
if ! python -c "import magenta_rt" 2>/dev/null; then
    print_error "Magenta RT package not installed. Please run 'pip install -e .' in the virtual environment."
    exit 1
fi

print_success "Magenta RT package installed"

# Check 3: JAX and GPU
print_status "Checking JAX and GPU availability..."
GPU_AVAILABLE=$(python -c "import jax; print(len([d for d in jax.devices() if d.platform == 'gpu']) > 0)" 2>/dev/null || echo "False")

if [ "$GPU_AVAILABLE" = "True" ]; then
    print_success "GPU detected and available for JAX"
    export MAGENTA_RT_DEVICE="gpu"
else
    print_warning "No GPU detected. Real Magenta RT requires GPU."
    print_warning "The system will fall back to mock mode."
    export MAGENTA_RT_DEVICE="gpu"  # Still try GPU, will fallback if needed
fi

# Check 4: Google Cloud credentials
print_status "Checking Google Cloud credentials..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_warning "Google Cloud CLI not found"
    print_status "Would you like to install it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Installing Google Cloud CLI..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux installation
            curl https://sdk.cloud.google.com | bash
            exec -l $SHELL
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS installation
            if command -v brew &> /dev/null; then
                brew install google-cloud-sdk
            else
                print_error "Homebrew not found. Please install Google Cloud CLI manually:"
                print_error "https://cloud.google.com/sdk/docs/install"
                exit 1
            fi
        else
            print_error "Unsupported OS. Please install Google Cloud CLI manually:"
            print_error "https://cloud.google.com/sdk/docs/install"
            exit 1
        fi
    else
        print_warning "Skipping Google Cloud CLI installation"
    fi
fi

# Check if authenticated
if command -v gcloud &> /dev/null; then
    if gcloud auth application-default print-access-token &> /dev/null; then
        print_success "Google Cloud credentials found"

        # Check if project is set
        CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
        if [ -z "$CURRENT_PROJECT" ]; then
            print_warning "No Google Cloud project set"
            print_status "Setting up a default project for Magenta RT (free)..."
            gcloud config set project magenta-rt-demo
            print_success "Project configured: magenta-rt-demo"
        else
            print_success "Using Google Cloud project: $CURRENT_PROJECT"
        fi

        CREDENTIALS_OK=true
    else
        print_warning "Google Cloud credentials not found"
        print_status "Would you like to authenticate now? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_status "Opening browser for authentication..."
            gcloud auth application-default login
            if gcloud auth application-default print-access-token &> /dev/null; then
                print_success "Authentication successful!"

                # Set up project
                print_status "Setting up a default project for Magenta RT (free)..."
                gcloud config set project magenta-rt-demo
                print_success "Project configured: magenta-rt-demo"

                CREDENTIALS_OK=true
            else
                print_error "Authentication failed"
                CREDENTIALS_OK=false
            fi
        else
            print_warning "Skipping authentication. System will use mock mode."
            CREDENTIALS_OK=false
        fi
    fi
else
    print_warning "Google Cloud CLI not available. System will use mock mode."
    CREDENTIALS_OK=false
fi

# Set environment variables
print_status "Setting up environment variables..."

# Create or update .env file
cat > .env << EOF
# Magenta RT Configuration
MAGENTA_RT_DEVICE=gpu
MAGENTA_RT_MODEL=large

# Set to 'true' to force mock mode for testing
# FORCE_MOCK_MODE=false
EOF

print_success "Environment variables configured in .env file"

# Summary
echo ""
echo "🎵 Setup Summary"
echo "================"

if [ "$CREDENTIALS_OK" = true ]; then
    print_success "✅ Real Magenta RT system should work!"
    print_status "Expected behavior: AI-generated music"
else
    print_warning "⚠️  Will use mock system (fake audio)"
    print_status "To enable real AI music later:"
    print_status "  1. Install Google Cloud CLI"
    print_status "  2. Run: gcloud auth application-default login"
    print_status "  3. Restart the application"
fi

echo ""
print_status "Configuration:"
print_status "  Device: $MAGENTA_RT_DEVICE"
print_status "  Model: large"
print_status "  GPU Available: $GPU_AVAILABLE"
print_status "  Credentials: ${CREDENTIALS_OK:-false}"

echo ""
print_status "You can now run: npm run dev"
print_status "The backend will automatically use the best available system."

echo ""
print_status "To check system status later: curl http://localhost:8000/api/health"
