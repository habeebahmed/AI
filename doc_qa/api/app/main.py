import shutil
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src import load_data,process_llm_response

app = FastAPI()

class Schema(BaseModel):
    question: str

qa_chain = None

@app.post('/uploadfile')
def upload_file(files: list[UploadFile]):
    global qa_chain
    try:
        for file in files:
            with open(f"/home/azureuser/doc_qa/data/{file.filename}", "wb") as f:
                shutil.copyfileobj(file.file,f)
        qa_chain = load_data()
        return JSONResponse(content = "File Uploaded successfully", status_code=200)
    except Exception:
        return JSONResponse(content="File upload was not successful", status_code=400)


@app.post('/ask')
def run_llm(data:Schema):
    global qa_chain
    try:
        user_input = data.question
        llm_repsonse = qa_chain(user_input)
        result, source = process_llm_response(llm_repsonse)
        return JSONResponse(content = {"Answer":result, "Source":source}, status_code = 200)
    except Exception:
        return JSONResponse(content="Could not process request, Please try again", status_code=500)
