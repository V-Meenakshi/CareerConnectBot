import streamlit as st
from streamlit_chat import message as st_message
from timeit import default_timer as timer
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_store import GraphStore
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import dotenv
import os
import google.generativeai as genai
import json
import traceback

dotenv.load_dotenv()
cypher_query = None
database_results = None

# API key for Google Gemini
GOOGLE_API_KEY = "your_apikey"

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# List available models to check what's accessible with your API key
try:
    available_models = [m.name for m in genai.list_models()]
    print("Available models:", available_models)
except Exception as e:
    print(f"Could not list models: {e}")
    available_models = []

# Use gemini-1.5-flash model as requested
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Using the flash model as requested
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    top_p=0.95,
    max_output_tokens=4096  # Increased token limit for more complete responses
)

# Neo4j configuration
neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_password"

# Improved Cypher generation prompt with fuzzy matching emphasis
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. IMPORTANT: Always use CASE-INSENSITIVE and FUZZY search for any properties. Use toLower() and CONTAINS for text matches
5. Never use relationships that are not mentioned in the given schema
6. When asked about jobs, interpret queries broadly and match any relevant properties
7. ALWAYS return ALL available properties for each job node including title, company, location, salary, description, job_url, and any other properties available
8. Set a LIMIT of at least 25 results to ensure we get comprehensive data
9. For partial matches like "data" should match "data science", "data engineer", etc.
10. IMPORTANT: Return ONLY the Cypher query without any prefixes like "Answer:" or explanations

schema: {schema}

Examples:
Question: Find software engineering jobs in Texas.
MATCH (j:Job) WHERE toLower(j.title) CONTAINS 'software' AND toLower(j.location) CONTAINS 'texas' RETURN j.title, j.company, j.location, j.salary, j.description, j.job_url, j.date_posted, j.job_type 

Question: jobs available in microsoft
MATCH (j:Job) WHERE toLower(j.company) CONTAINS 'microsoft' RETURN j.title, j.company, j.location, j.salary, j.description, j.job_url, j.date_posted, j.job_type 

Question: data science jobs
MATCH (j:Job) WHERE toLower(j.title) CONTAINS 'data' OR toLower(j.title) CONTAINS 'science' OR toLower(j.description) CONTAINS 'data science' RETURN j.title, j.company, j.location, j.salary, j.description, j.job_url, j.date_posted, j.job_type 

Question: events that are located at bangalore
MATCH (e:Event)-[:HAS_LOCATION]->(l:Location)
OPTIONAL MATCH (e)-[:HAS_TAG]->(t:Tag)
WHERE toLower(l.name) CONTAINS 'bangalore'
RETURN e.name, e.mode, l.name AS location, e.date_range, e.time_range, e.id, collect(t.name) AS tags

Question: {question}
Cypher Query:
"""

cypher_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema", "question"]
)

# Improved QA template to handle empty results better
CYPHER_QA_TEMPLATE = """
You are a helpful job search assistant that provides clear, structured responses based solely on database results.

IMPORTANT INSTRUCTIONS:
1. If the database returns any jobs, present ALL of them with ALL available information (title, company, location, salary, etc.)
2. If NO jobs are found in the database results, DO NOT make up generic advice or ask for more details. Instead, clearly state you found no matches and suggest a broader search or alternative terms.
3. Present job information in a clear, easy-to-read format with proper spacing between jobs
4. Always include available salary information
5. Do not apologize or be overly conversational
6. Be direct and to the point - focus on presenting the job information clearly
7. If you're showing multiple jobs, number them for clarity

Here is the database information you must use:
{context}

The user's question was: {question}

Respond with ONLY the job information from the database or a clear statement that no matching jobs were found. Do not add generic advice if no results are found.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

# Make sure Neo4jGraph is registered as a GraphStore
if not issubclass(Neo4jGraph, GraphStore):
    # Create a new class that inherits from both Neo4jGraph and GraphStore
    class Neo4jGraphWithStore(Neo4jGraph, GraphStore):
        pass
    
    # Replace the original Neo4jGraph with our new class
    Neo4jGraph = Neo4jGraphWithStore

def test_neo4j_connection():
    """Test the Neo4j connection and return connection status"""
    try:
        neo4j_graph = Neo4jGraph(
            url=neo4j_url, 
            username=neo4j_user, 
            password=neo4j_password
        )
        # Try a simple query to verify connection
        result = neo4j_graph.query("MATCH (n) RETURN count(n) as count LIMIT 1")
        return True, f"Connected to Neo4j. Database contains {result[0]['count']} nodes."
    except Exception as e:
        return False, f"Failed to connect to Neo4j: {str(e)}"

def preprocess_cypher_query(cypher_query):
    """Clean up any issues with the generated Cypher query"""
    if not cypher_query:
        return cypher_query
        
    # Remove "Answer:" prefix if present
    if cypher_query.startswith("Answer:"):
        cypher_query = cypher_query[7:].strip()
    
    # Remove any other potential prefixes
    prefixes = ["Cypher:", "Cypher Query:", "Query:"]
    for prefix in prefixes:
        if cypher_query.startswith(prefix):
            cypher_query = cypher_query[len(prefix):].strip()
    
    return cypher_query

def query_graph(user_input):
    try:
        # Initialize Neo4jGraph connection
        neo4j_graph = Neo4jGraph(
            url=neo4j_url, 
            username=neo4j_user, 
            password=neo4j_password,
            database="neo4j"  # Explicitly specify the database name
        )
        
        print("Neo4jGraph successfully initialized!")
        
        # Verify that it's a GraphStore instance
        if not isinstance(neo4j_graph, GraphStore):
            raise TypeError("neo4j_graph is not a valid GraphStore instance after patching!")
            
        # Create GraphCypherQAChain using Neo4jGraph
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=neo4j_graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            top_k=30  # Increase from default to get more results
        )

        # Run the chain with user input
        result = chain(user_input)
        
        # Extract intermediate steps safely
        cypher_query = "No query found"
        database_results = "No results found"
        
        if isinstance(result, dict) and "intermediate_steps" in result:
            intermediate_steps = result["intermediate_steps"]
            if isinstance(intermediate_steps, list) and len(intermediate_steps) > 0:
                raw_query = intermediate_steps[0].get("query", "No query found")
                # Clean up the query to remove prefixes
                cypher_query = preprocess_cypher_query(raw_query)
                
                # Re-execute the query with the cleaned version if needed
                if raw_query != cypher_query and cypher_query != "No query found":
                    try:
                        print("Re-executing with cleaned query:", cypher_query)
                        results = neo4j_graph.query(cypher_query)
                        intermediate_steps[1]["context"] = results
                        database_results = results
                    except Exception as e:
                        print(f"Error with cleaned query: {e}")
                        # Fall back to original results
                        if len(intermediate_steps) > 1:
                            database_results = intermediate_steps[1].get("context", "No results found")
                else:
                    if len(intermediate_steps) > 1:
                        database_results = intermediate_steps[1].get("context", "No results found")

        # Check if the result is empty and generate a better response
        if isinstance(database_results, list) and len(database_results) == 0:
            final_answer = f"I searched for jobs matching '{user_input}' but found no results in the database. Try broadening your search terms or using different keywords."
            # Update the result with our custom answer
            result["result"] = final_answer

        print("Generated Cypher Query:", cypher_query)
        print("Database Results (preview):", str(database_results)[:500] + "..." if len(str(database_results)) > 500 else database_results)

        return {
            "result": result.get("result", "No answer generated"),
            "cypher_query": cypher_query,
            "database_results": database_results,
            "intermediate_steps": result.get("intermediate_steps", [])
        }

    except Exception as e:
        print(f"Error in query_graph: {e}")
        traceback.print_exc()
        return {
            "result": f"Failed to process the query: {str(e)}",
            "cypher_query": None,
            "database_results": None,
            "intermediate_steps": []
        }

def direct_cypher_query(query_text):
    """Execute a direct Cypher query against Neo4j and return the results"""
    try:
        neo4j_graph = Neo4jGraph(
            url=neo4j_url, 
            username=neo4j_user, 
            password=neo4j_password
        )
        
        # Clean the query first
        clean_query = preprocess_cypher_query(query_text)
        results = neo4j_graph.query(clean_query)
        return True, results
    except Exception as e:
        return False, f"Query error: {str(e)}"

def try_alternative_gemini_models(user_input):
    """Attempt to use different Gemini models if the primary one fails"""
    # List of models to try in order of preference
    models_to_try = ["gemini-1.0-pro", "gemini-pro-latest", "gemini-1.0-pro-latest"]
    
    for model_name in models_to_try:
        try:
            print(f"Trying alternative model: {model_name}")
            alt_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
                max_output_tokens=4096  # Increased token limit
            )
            
            # Initialize Neo4jGraph connection
            neo4j_graph = Neo4jGraph(
                url=neo4j_url, 
                username=neo4j_user, 
                password=neo4j_password
            )
            
            # Create GraphCypherQAChain using the alternative model
            chain = GraphCypherQAChain.from_llm(
                llm=alt_llm,
                graph=neo4j_graph,
                verbose=True,
                return_intermediate_steps=True,
                cypher_prompt=cypher_prompt,
                qa_prompt=qa_prompt,
                top_k=30  # Increase from default
            )
            
            # Run the chain with user input
            result = chain(user_input)
            
            # Process results
            cypher_query = "No query found"
            database_results = "No results found"
            
            if isinstance(result, dict) and "intermediate_steps" in result:
                intermediate_steps = result["intermediate_steps"]
                if isinstance(intermediate_steps, list) and len(intermediate_steps) > 0:
                    raw_query = intermediate_steps[0].get("query", "No query found")
                    # Clean up the query to remove prefixes
                    cypher_query = preprocess_cypher_query(raw_query)
                    
                    # Re-execute the query with the cleaned version if needed
                    if raw_query != cypher_query and cypher_query != "No query found":
                        try:
                            print("Re-executing with cleaned query:", cypher_query)
                            results = neo4j_graph.query(cypher_query)
                            intermediate_steps[1]["context"] = results
                            database_results = results
                        except Exception as e:
                            print(f"Error with cleaned query: {e}")
                            # Fall back to original results
                            if len(intermediate_steps) > 1:
                                database_results = intermediate_steps[1].get("context", "No results found")
                    else:
                        if len(intermediate_steps) > 1:
                            database_results = intermediate_steps[1].get("context", "No results found")
            
            # Check if the result is empty and generate a better response
            if isinstance(database_results, list) and len(database_results) == 0:
                final_answer = f"I searched for jobs matching '{user_input}' but found no results in the database. Try broadening your search terms or using different keywords."
                # Update the result with our custom answer
                result["result"] = final_answer
                    
            return {
                "result": result.get("result", f"No answer generated (using alternative model: {model_name})"),
                "cypher_query": cypher_query,
                "database_results": database_results,
                "intermediate_steps": result.get("intermediate_steps", []),
                "model_used": model_name
            }
                
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue
    
    # If all models fail, return an error
    return {
        "result": "All Gemini models failed. Please check your API key or try again later.",
        "cypher_query": None,
        "database_results": None,
        "intermediate_steps": [],
        "model_used": None
    }

# Fallback direct search function when LLM approaches fail
def basic_job_search(search_terms):
    """Perform a basic search with direct Cypher when other methods fail"""
    try:
        # Create a simple query that searches for terms in title, company, and description
        terms = search_terms.lower().split()
        where_clauses = []
        
        for term in terms:
            if term not in ['jobs', 'job', 'positions', 'role', 'roles', 'available', 'for', 'in', 'at', 'the', 'and', 'or']:
                where_clauses.append(f"toLower(j.title) CONTAINS '{term}' OR toLower(j.company) CONTAINS '{term}' OR toLower(j.location) CONTAINS '{term}'")
        
        # If no valid search terms, search everything
        if not where_clauses:
            cypher_query = """
            MATCH (j:Job)
            RETURN j.title, j.company, j.location, j.salary, j.description, j.job_url, j.date_posted, j.job_type
            LIMIT 25
            """
        else:
            where_statement = " OR ".join(where_clauses)
            cypher_query = f"""
            MATCH (j:Job)
            WHERE {where_statement}
            RETURN j.title, j.company, j.location, j.salary, j.description, j.job_url, j.date_posted, j.job_type
            LIMIT 25
            """
        
        # Execute the query
        neo4j_graph = Neo4jGraph(
            url=neo4j_url, 
            username=neo4j_user, 
            password=neo4j_password
        )
        
        results = neo4j_graph.query(cypher_query)
        return True, cypher_query, results
    except Exception as e:
        print(f"Basic search error: {e}")
        return False, str(e), []

# Initialize Streamlit UI with a clean, chat-like interface
st.set_page_config(layout="wide", page_title="Job Finder Bot", page_icon="💼")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []
if "raw_results" not in st.session_state:
    st.session_state.raw_results = []

# Clean, minimalist header
st.header("💼 Job Finder Bot")

# Test connection and display status in sidebar
connection_status, message = test_neo4j_connection()
if connection_status:
    st.sidebar.success(message)
else:
    st.sidebar.error(message)

# Advanced options in sidebar (hidden by default)
with st.sidebar:
    st.subheader("Advanced Options")
    with st.expander("Direct Cypher Query", expanded=False):
        cypher_example = """MATCH (j:Job) 
WHERE toLower(j.title) CONTAINS 'software' AND toLower(j.title) CONTAINS 'intern'
RETURN j.title, j.company, j.location, j.salary, j.job_url
LIMIT 20"""
        
        direct_query = st.text_area("Enter Cypher Query", value=cypher_example, height=200)
        
        if st.button("Execute Query"):
            with st.spinner("Executing query..."):
                success, query_result = direct_cypher_query(direct_query)
                
                if success:
                    st.success(f"Query executed successfully. Found {len(query_result)} results.")
                    st.json(query_result)
                else:
                    st.error(query_result)  # Display error message
    
    with st.expander("View Database Schema", expanded=False):
        if st.button("Show Database Schema"):
            with st.spinner("Retrieving schema..."):
                success, schema_result = direct_cypher_query("""
                CALL db.schema.visualization()
                """)
                
                if success:
                    st.json(schema_result)
                else:
                    st.error(schema_result)

# Main chat interface
chat_container = st.container()

# Chat history display
with chat_container:
    if st.session_state.system_msgs:
        for i in range(len(st.session_state.system_msgs)):
            st_message(st.session_state.user_msgs[i], is_user=True, key=f"{i}_user")
            
            # Display message with expandable details
            message_container = st.container()
            with message_container:
                st_message(st.session_state.system_msgs[i], key=f"{i}_assistant")
                
                # Only show details if we have them
                if i < len(st.session_state.raw_results) and st.session_state.raw_results[i]:
                    result = st.session_state.raw_results[i]
                    with st.expander("View Query Details", expanded=False):
                        if result.get("cypher_query"):
                            st.subheader("Cypher Query")
                            st.code(result["cypher_query"], language="cypher")
                        
                        if result.get("database_results"):
                            st.subheader("Database Results")
                            try:
                                st.json(result["database_results"])
                            except:
                                st.text_area("Raw Results", str(result["database_results"]), height=200)

# Input area at the bottom
st.markdown("---")
user_input = st.text_input("Ask about jobs (e.g., 'Find software engineering jobs in New York')", key="user_query")

# Search button
if st.button("Search Jobs", type="primary") and user_input:
    with st.spinner("Searching for jobs..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            # Try LLM approach first
            result = query_graph(user_input)
            
            # If the LLM approach fails, try alternatives
            if "Failed to process the query" in result["result"]:
                alt_result = try_alternative_gemini_models(user_input)
                
                # Use alternative result if it succeeded
                if not alt_result["result"].startswith("All Gemini models failed"):
                    result = alt_result
                    with st.sidebar:
                        st.success(f"Used alternative model: {alt_result.get('model_used')}")
                else:
                    # If all LLM approaches fail, try direct query as last resort
                    with st.sidebar:
                        st.warning("All LLM models failed. Trying direct search...")
                    
                    success, cypher_query, search_results = basic_job_search(user_input)
                    if success and search_results:
                        # Format the results in a readable way
                        formatted_results = []
                        for job in search_results:
                            job_info = []
                            for key, value in job.items():
                                if value and str(value).strip():
                                    job_info.append(f"**{key.replace('j.', '')}**: {value}")
                            formatted_results.append("\n".join(job_info))
                        
                        if formatted_results:
                            result_text = f"Here are jobs matching '{user_input}':\n\n"
                            for i, job_text in enumerate(formatted_results[:20], 1):
                                result_text += f"**Job {i}**\n{job_text}\n\n"
                        else:
                            result_text = f"I searched for '{user_input}' but couldn't find any matching jobs."
                        
                        result = {
                            "result": result_text,
                            "cypher_query": cypher_query,
                            "database_results": search_results,
                            "intermediate_steps": []
                        }
            
            # Store results and update UI
            st.session_state.system_msgs.append(result["result"])
            st.session_state.raw_results.append(result)
            
            # Force UI refresh - without experimental_rerun
            st_message(user_input, is_user=True, key="new_user")
            st_message(result["result"], key="new_assistant")
            
            with st.expander("View Query Details", expanded=False):
                if result.get("cypher_query"):
                    st.subheader("Cypher Query")
                    st.code(result["cypher_query"], language="cypher")
                
                if result.get("database_results"):
                    st.subheader("Database Results")
                    try:
                        st.json(result["database_results"])
                    except:
                        st.text_area("Raw Results", str(result["database_results"]), height=200)
                        
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            st.error(error_msg)
            st.session_state.system_msgs.append(error_msg)
            st.session_state.raw_results.append({"result": error_msg})
            traceback.print_exc()
            
        with st.sidebar:
            st.info(f"Time taken: {timer() - start:.2f}s")

# Display a helpful message when the app starts
if not st.session_state.user_msgs:
    st.info("👋 Welcome to the Job Finder Bot! Ask me about job listings by typing in the search box below.")
