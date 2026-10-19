#!/bin/bash
# LaTeX Workshop wrapper script for TeX Live 2022
export PATH="/home/PengYanghan/texlive/2022/bin/x86_64-linux:$PATH"
export MANPATH="/home/PengYanghan/texlive/2022/texmf-dist/doc/man:$MANPATH"
exec "/home/PengYanghan/texlive/2022/bin/x86_64-linux/latexmk" "$@"
