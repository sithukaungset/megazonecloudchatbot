# Chatbot Data Interpreter

This application is a chatbot designed to read data from various sources, process it, and answer user queries based on the loaded data and a pre-trained model. It supports data input from formats like PDFs, Excel files, and TXT files.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Modules](#modules)
4. [Key Classes & Methods](#key-classes-and-methods)
5. [Flow](#flow)
6. [Notes](#notes)
7. [Schematic Diagram](#schematic-diagram)
8. [Contributing](#contributing)
9. [License](#license)

## Features

- Load environment variables from a `.env` file.
- User-friendly web interface for file uploads and querying.
- Read and process data from TXT, CSV, PDF, and Excel files.
- Interact with OpenAI for advanced query processing.
- Store chat history in a SQLite database.

## Architecture

The code comprises multiple modules, each responsible for specific functionalities, such as reading files, processing data, or interacting with APIs.

## Modules

- **dotenv**: Load environment variables from a `.env` file.
- **os**: Interact with the OS, primarily for environment variables.
- **pandas**: Data manipulation and analysis.
- **streamlit**: Web app framework for data scripts.
- **PyPDF2, fitz (PyMuPDF)**: Interact with PDF files.
- **tiktoken, openai**: Handle interactions with OpenAI.
- **pytesseract**: OCR tool to extract text from images.
- **pdfminer**: Extract text from PDFs.
- **pdf2image**: Convert PDF pages to images for OCR.
- **re**: Python's regular expression library.
- **sqlite3**: Lightweight disk-based database.

## Key Classes and Methods

- **TabularDataProcessor**: Process tabular data formats. Contains methods for preprocessing and converting data into sentences.
- **ocr_pdf()**, **process_pdf()**: Process PDFs using OCR and extract mathematical content.
- **translate()**: Placeholder function for text translation.
- **main()**: Creates the UI, processes files, and answers user queries.

## Flow

1. **Setup**: Load environment variables.
2. **UI Creation with Streamlit**: Page setup, model selection, chat mode toggle.
   - **Natural Chat Mode**: Fetch chat history, get user input, display and store response.
   - **File Upload Mode**: Allow file uploads, read and process files, answer queries.
  
## Notes

- Some imports in the code are commented out.
- The `translate()` function is currently a stub.
- Placeholder code hints at potential future features.
- The database connection doesn't have a clear closing mechanism.

## Schematic Diagram

For visual representation, consider using tools like Lucidchart, draw.io, or Microsoft Visio. This README doesn't provide actual diagrams but offers guidance on creating them.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss the proposed change.

## License

Please refer to the project's license file for information.
