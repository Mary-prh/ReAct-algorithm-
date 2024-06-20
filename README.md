# ReAct-algorithm

Implementation of the ReAct algorithm using the LangChain framework.

It sets up an agent capable of answering questions by reasoning through a defined process, utilizing specified tools, and then parsing the output to arrive at a final answer.

## Scripts

### main_count.py

This script sets up a tool that counts the number of alphabetic characters in a given text after removing non-alphabetic characters.

### main_facts.py

This script sets up a tool that fetches a random fact from an online API and returns it to the user.

### main_currency.py

This script sets up a tool that performs real-time currency conversion. It takes user input in the format 'amount from_currency to to_currency' (e.g., '100 USD to CAD'), fetches the current exchange rate using the ExchangeRate-API, and returns the converted amount.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/react-algorithm.git
   cd react-algorithm
