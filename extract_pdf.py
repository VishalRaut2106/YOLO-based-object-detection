import sys

try:
    import fitz # PyMuPDF
    doc = fitz.open(sys.argv[1])
    text = ""
    for page in doc:
        text += page.get_text()
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(text)
    print("Extracted with PyMuPDF")
except ImportError:
    try:
        from pypdf import PdfReader
        reader = PdfReader(sys.argv[1])
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted with pypdf")
    except ImportError:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(sys.argv[1])
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            with open(sys.argv[2], "w", encoding="utf-8") as f:
                f.write(text)
            print("Extracted with PyPDF2")
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
            import fitz
            doc = fitz.open(sys.argv[1])
            text = ""
            for page in doc:
                text += page.get_text()
            with open(sys.argv[2], "w", encoding="utf-8") as f:
                f.write(text)
            print("Installed and extracted with PyMuPDF")
