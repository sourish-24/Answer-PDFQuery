import * as dotenv from 'dotenv';
dotenv.config();

import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

import { GoogleGenAI } from "@google/genai";
const ai = new GoogleGenAI({});  //automatically reads GEMINI_API_KEY from .env
const History = []


async function chatting(query) {
    //convert user query into vector
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });

    const queryVector = await embeddings.embedQuery(query);

    //connect to pineconeDB
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
        topK: 3,
        vector: queryVector,
        includeMetadata: true,
    });

    //console.log(searchResults);

    //take topK matches and combine match.metadata.text into a context for RAG
    const context = searchResults.matches
        .map(match => match.metadata.text)
        .join("\n\n---\n\n");


    //gemini api call
    History.push({
        role: 'user',
        parts: [{ text: query }]
    })

    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: History,
        config: {
            systemInstruction: `You have to behave like an Insurance cover expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    You will read the user's question and compare it with the context provided, you will respond only with 
    YES or NO, along with reasoning from the context.
    Assume that whoever the user query is taking about is an insured person, they are just checking about 
    the validity of their insurance cover.
    If you need more details, you may ask the user for specific details. 
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and logical.
    Sample Query "46M, knee surgery, Pune, 3-month policy"
    Sample Response "Yes, knee surgery is covered under the policy."
      
      Context: ${context}
      `,
        },
    });


    History.push({
        role: 'model',
        parts: [{ text: response.text }]
    })

    console.log("\n");
    console.log(response.text);
}


async function main() {
    const userQuery = readlineSync.question("Ask me anything--> ");
    await chatting(userQuery);
    main();
}


main();