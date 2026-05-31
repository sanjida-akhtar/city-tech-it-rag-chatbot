from langchain_community.document_loaders import SeleniumURLLoader
import json
import os

# loading pages
selenium_loader = SeleniumURLLoader(
    urls= ["https://www.citytechit.com/",
            "https://www.citytechit.com/index.php#about",
            "https://www.citytechit.com/index.php#features",
            "https://www.citytechit.com/service.php",
            "https://www.citytechit.com/nsda.php",
            "https://www.citytechit.com/privat_course.php",
            "https://www.citytechit.com/index.php#contact"],
    headless =True,
    browser = "chrome"
)
# convert to document object
documents = selenium_loader.load()
print(documents[0].page_content)

# make a directory
os.makedirs("../data/raw", exist_ok=True)

# Convert document object into a list
docs = []

for doc in documents:
    docs.append({
        "content": doc.page_content,
        "metadata": doc.metadata
    })

# save docs into a json file
with open("../data/raw/docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=4)
