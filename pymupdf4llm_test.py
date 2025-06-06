from langchain_pymupdf4llm import PyMuPDF4LLMLoader

loader = PyMuPDF4LLMLoader("A_TravelPolicyBot_Original.pdf")
docs = loader.load()

print(len(docs))
print(docs[0].page_content[:500])
