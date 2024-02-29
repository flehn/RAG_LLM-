import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import FAISS
import nest_asyncio
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import transformers

##################################################################
# Model
#################################################################

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
#https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

#model to convert text into embeddings
embedding_model='sentence-transformers/all-mpnet-base-v2'

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)


# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device == 'cuda':

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
else: 
    bnb_config = None






# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Model
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             quantization_config=bnb_config,
                                             )




nest_asyncio.apply()


def read_articles(articles: list):

    # Scrapes all articles in articles 
    loader = AsyncChromiumLoader(articles)
    docs = loader.load()

    # Converts HTML to plain text 
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=100, 
                                        chunk_overlap=20)
    chunked_documents = text_splitter.split_documents(docs_transformed)

    # Load chunked documents into the FAISS index
    db = FAISS.from_documents(chunked_documents, 
                            HuggingFaceEmbeddings(model_name=embedding_model))

    return db




def predict(query:str, db):


    text_generation_pipeline = transformers.pipeline(
    model=model,
    do_sample=True,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
    )

    prompt_template = """
    ### [INST] 
    Instruction: Answer the question based on your knowledge. Here is context to help:

    {context}

    ### QUESTION:
    {question} 

    [/INST]
    """

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain 
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    #retrieve context
    # Connect query to FAISS index using a retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )

    rag_chain = ( 
                {"context": retriever, "question": RunnablePassthrough()}
                    | llm_chain
                )
    
    return rag_chain.invoke(query)