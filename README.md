# Job Finder Bot

## Overview
Job Finder Bot is an intelligent job search application that leverages natural language processing to help users find job listings using conversational queries. The application connects to a Neo4j graph database and uses Google's Gemini AI models to interpret user queries, convert them to Cypher queries, and present job results in a user-friendly format.

## Features
- **Natural Language Job Search**: Ask for jobs in plain English (e.g., "Find software engineering jobs in New York")
- **Graph Database Integration**: Utilizes Neo4j's graph capabilities for efficient job data storage and retrieval
- **AI-Powered Query Processing**: Converts natural language to Cypher queries using Google's Gemini models
- **Interactive Chat Interface**: Clean, user-friendly UI built with Streamlit
- **Advanced Options**:
  - Execute direct Cypher queries for power users
  - View database schema
  - Examine query details and raw results
- **Multiple Fallback Methods**: Uses alternative models or direct search if primary approach fails

## Technologies
- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **Neo4j**: Graph database for storing job information
- **LangChain**: Framework for building applications with LLMs
- **Google Generative AI (Gemini)**: AI models for natural language understanding
- **GraphCypherQAChain**: For converting questions to Cypher queries
- **Streamlit Chat**: For chat interface components

## Setup & Installation

### Prerequisites
- Python 3.8+
- Neo4j Database (local or remote)
- Google AI API key

### Environment Setup
1. Clone this repository
2. Install dependencies:
   ```
   pip install streamlit langchain langchain-neo4j langchain-google-genai streamlit-chat google-generativeai python-dotenv
   ```
3. Create a `.env` file with the following variables:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

### Neo4j Database Setup
1. Install and run Neo4j
2. Create a database for job listings
3. Import job data (format should include nodes with the Job label and appropriate properties)

## Usage

### Running the Application
```
streamlit run app.py
```

### Search Examples
- "Find software engineering jobs in Texas"
- "Data science positions at Google"
- "Remote marketing jobs with salary above 100k"
- "Entry level engineer positions in California"

### Advanced Usage
- Use the sidebar to access direct Cypher query execution
- View database schema
- Expand query details to see the generated Cypher and raw results

## How It Works
1. User enters a natural language job query
2. The application uses Gemini AI to convert the query to a Neo4j Cypher query
3. The Cypher query is executed against the Neo4j database
4. Results are processed and presented to the user in a readable format
5. If the primary approach fails, the system tries alternative models or a basic search


## Future Improvements
- Add user authentication
- Implement job application tracking
- Add filters for job search (salary range, job type, etc.)
- Implement job recommendations based on user history
- Add data visualization for job market trends

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Include your license information here]
