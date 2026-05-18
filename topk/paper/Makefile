TEXDIR   = /home/mukunoki/local/bin/x86_64-linux
PLATEX   = $(TEXDIR)/platex
DVIPDFMX = $(TEXDIR)/dvipdfmx
PBIBTEX  = $(TEXDIR)/pbibtex
PATH    := $(CURDIR):$(PATH)

TARGET = new
SRC    = $(TARGET).tex
TABLES = verilog_eval_feature_table.tex \
         verilog_eval_correlation_table.tex \
         verilog_ugt_overall_full_table.tex \
         verilog_ugt_struct_full_table.tex
FIGURES = verilog_hybrid_struct_bar.eps \
          verilog_hybrid_per_problem.eps

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(SRC) $(TABLES) $(FIGURES) references.bib
	$(PLATEX) $(SRC)
	-$(PBIBTEX) $(TARGET)
	$(PLATEX) $(SRC)
	$(PLATEX) $(SRC)
	$(DVIPDFMX) $(TARGET).dvi

clean:
	rm -f $(TARGET).pdf $(TARGET).aux $(TARGET).log $(TARGET).bbl \
	      $(TARGET).blg $(TARGET).out $(TARGET).dvi $(TARGET).synctex.gz
