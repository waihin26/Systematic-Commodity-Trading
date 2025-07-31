#!/bin/bash

# Gold RSI Strategy Report Compilation Script
# This script compiles the LaTeX report into a PDF

echo "=== Compiling Gold RSI Strategy Report ==="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install a LaTeX distribution like:"
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu: sudo apt-get install texlive-full"
    echo "  Windows: Install MiKTeX or TeX Live"
    exit 1
fi

# Compile the LaTeX document (run twice for proper cross-references)
echo "Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode gold_rsi_strategy_report.tex

echo "Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode gold_rsi_strategy_report.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.toc *.out *.fdb_latexmk *.fls

# Check if PDF was created successfully
if [ -f "gold_rsi_strategy_report.pdf" ]; then
    echo "✅ Success! Report generated: gold_rsi_strategy_report.pdf"
    echo "📊 The report is ready for presentation to portfolio managers."
    
    # Get file size for verification
    size=$(ls -lh gold_rsi_strategy_report.pdf | awk '{print $5}')
    echo "📄 File size: $size"
    
    # Try to open the PDF (macOS only)
    if command -v open &> /dev/null; then
        echo "🔍 Opening PDF..."
        open gold_rsi_strategy_report.pdf
    fi
else
    echo "❌ Error: PDF compilation failed. Check the log files for details."
    exit 1
fi
