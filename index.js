//load pdf
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import * as dotenv from 'dotenv';
dotenv.config();
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

async function indexDocument() {
    const pdfLoader = new PDFLoader('./WellBaby.pdf');
    const rawDocs = await pdfLoader.load();


    //chunking
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);


    //vector embedding
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    console.log("Embedding done")


    //database
    //initialize pinecone client
    const pinecone = new Pinecone();  //automatically picks up PINECONE_API_KEY from .env
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);


    //langchain (chunking, embedding, database)
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });
    console.log("Data Stored in Database")
}

indexDocument();