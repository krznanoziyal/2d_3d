import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Read text from the image
result = reader.readtext('test.jpg')

# Extract only the text and dimensions
filtered_text = []
for item in result:
    text = item[1]
    if any(char.isdigit() for char in text) or text.isalpha():  # Check if it's a dimension or a label
        filtered_text.append(text)

# Print the extracted text and dimensions
for text in filtered_text:
    print(text)