import os
from langchain_text_splitters import MarkdownHeaderTextSplitter

DEFAULT_DATA_PATH = "scrapping/output_UQAC_Website"



# All markdown files are stored in the scrapping/output_UQAC_Website directory
documents = [doc for doc in os.listdir(DEFAULT_DATA_PATH) if doc.endswith('.md')]

print(f"Found {len(documents)} markdown files in {DEFAULT_DATA_PATH}.")
# print("Example of a document:")
# print(open(os.path.join(DEFAULT_DATA_PATH, documents[0])).read()[:200])

doc1 = documents[0]

with open(os.path.join(DEFAULT_DATA_PATH, doc1), 'r') as file:
    doc1 = file.read()

# print(doc1)
print('\n\n')

markdown_document = doc1

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# Char-level splits
from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split
splits = text_splitter.split_documents(md_header_splits)
splits


print(len(splits))

print(splits[1], '\nMMmmmmmmmmmmmmm\n')
print(splits[2], '\nMMmmmmmmmmmmmmm\n')
print(splits[3], '\nMMmmmmmmmmmmmmm\n')


