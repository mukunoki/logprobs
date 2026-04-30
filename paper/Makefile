TEXDIR   = /home/mukunoki/local/bin/x86_64-linux
PLATEX   = $(TEXDIR)/platex
DVIPDFMX = $(TEXDIR)/dvipdfmx
PBIBTEX  = $(TEXDIR)/pbibtex
PATH    := $(CURDIR):$(PATH)

TARGET = cbba_paper
SRC    = $(TARGET).tex

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(SRC)
	$(PLATEX) $(SRC)
	-$(PBIBTEX) $(TARGET)
	$(PLATEX) $(SRC)
	$(PLATEX) $(SRC)
	$(DVIPDFMX) $(TARGET).dvi

clean:
	rm -f $(TARGET).pdf $(TARGET).aux $(TARGET).log $(TARGET).bbl \
	      $(TARGET).blg $(TARGET).out $(TARGET).dvi $(TARGET).synctex.gz
