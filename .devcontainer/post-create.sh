#!/bin/bash
# =============================================================================
# Post-create script for SpecAgent dev container
# =============================================================================
# This script runs after the dev container is created.
# It installs project dependencies and sets up the development environment.
# =============================================================================

set -e

echo "🚀 Setting up SpecAgent development environment..."

# -----------------------------------------------------------------------------
# Install Claude Code CLI
# -----------------------------------------------------------------------------
echo "📦 Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Verify installation
if command -v claude &> /dev/null; then
    echo "✅ Claude Code installed: $(claude --version)"
else
    echo "⚠️  Claude Code installation may have failed. Install manually with: npm install -g @anthropic-ai/claude-code"
fi

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
echo "📦 Installing Python dependencies..."
cd /workspaces/specagent

# Install in development mode with all extras
pip install -e ".[dev,eval,ui]" --quiet

# Verify installation
if python -c "import specagent" 2>/dev/null; then
    echo "✅ SpecAgent package installed"
else
    echo "⚠️  SpecAgent import failed. Check installation."
fi

# -----------------------------------------------------------------------------
# Set up pre-commit hooks (optional)
# -----------------------------------------------------------------------------
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
fi

# -----------------------------------------------------------------------------
# Create .env file if it doesn't exist
# -----------------------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Remember to add your HF_API_KEY to .env"
fi

# -----------------------------------------------------------------------------
# Create data directories
# -----------------------------------------------------------------------------
echo "📁 Ensuring data directories exist..."
mkdir -p data/raw data/processed data/index

# -----------------------------------------------------------------------------
# Git configuration
# -----------------------------------------------------------------------------
echo "🔧 Configuring git..."
git config --global core.editor "code --wait"
git config --global init.defaultBranch main

# -----------------------------------------------------------------------------
# Print helpful information
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "✅ SpecAgent development environment ready!"
echo "=============================================="
echo ""
echo "Quick start commands:"
echo "  specagent --help          # CLI help"
echo "  specagent serve           # Start API server"
echo "  pytest                    # Run tests"
echo "  ruff check src/           # Lint code"
echo "  mypy src/specagent        # Type check"
echo ""
echo "Claude Code commands:"
echo "  claude                    # Start Claude Code"
echo "  claude --help             # Claude Code help"
echo ""
echo "⚠️  Don't forget to:"
echo "  1. Add your HF_API_KEY to .env"
echo "  2. Run 'huggingface-cli login' if downloading datasets"
echo ""
