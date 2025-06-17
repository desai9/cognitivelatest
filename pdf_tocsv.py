#!/usr/bin/env python3
"""
Enhanced PDF Bank Statement to CSV Converter
Version: 2.0.0
Last Updated: 2025-06-15
Author: Gowtham619
"""

import PyPDF2
import ollama
import csv
import argparse
import sys
import os
import json
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from tqdm import tqdm
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor.log'),
        logging.StreamHandler()
    ]
)

class BankStatementProcessor:
    def __init__(self, model="gemma2:2b"):
        self.model = model
        self.success_count = 0
        self.failure_count = 0
        self.failed_files = []
        self.start_time = datetime.utcnow()
        logging.info(f"Initializing BankStatementProcessor with model: {model}")
    
    def extract_text_from_pdf(self, pdf_path: str, password: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
        """Extract text content from PDF file with password support and retries"""
        for attempt in range(max_retries):
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    if pdf_reader.is_encrypted:
                        if not password:
                            raise ValueError(f"PDF is encrypted but no password provided: {pdf_path}")
                        try:
                            pdf_reader.decrypt(password)
                        except:
                            if attempt == max_retries - 1:
                                raise ValueError(f"Invalid password for {pdf_path}")
                            continue
                    
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Error reading PDF {pdf_path}: {str(e)}")
                    return None
                time.sleep(1)  # Add delay between retries
        return None

    def create_parsing_prompt(self, pdf_text: str) -> str:
        """Create a prompt for the LLM to parse bank statement"""
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        prompt = f"""
You are a financial data extraction expert. Current UTC time is {current_time}.
Analyze the following bank statement text and extract all transaction data.

For each transaction, provide the output in this exact CSV format:
date,description,amount

Rules:
1. Date format: YYYY-MM-DD
2. Description: Clean transaction description (remove extra spaces, special characters)
3. Amount: Positive for credits/deposits, negative for debits/withdrawals
4. Only include actual transactions, not headers or summaries
5. Skip any balances, fees summaries, or non-transaction lines
6. If amount shows as debit or withdrawal, make it negative
7. If unclear whether debit/credit, use context clues from description
8. Remove any special characters from descriptions except commas in numbers
9. For missing dates, use the previous valid transaction date
10. Skip lines that don't represent actual transactions

Bank statement text:
{pdf_text}

Provide ONLY the CSV data with header row, no other text or explanations:
"""
        return prompt

    def query_ollama(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Send query to Ollama with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt
                )
                return response['response']
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Error querying Ollama (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    return None
                logging.warning(f"Ollama query failed (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)  # Add small delay between retries
        return None

    def clean_csv_response(self, response: str) -> str:
        """Clean and validate the CSV response from LLM"""
        lines = response.strip().split('\n')
        cleaned_lines = []
        current_date = None
        
        # Find the header line
        header_found = False
        for line in lines:
            if 'date,description,amount' in line.lower():
                cleaned_lines.append('date,description,amount')
                header_found = True
                continue
            
            if header_found and line.strip():
                # Basic validation - should have exactly 2 commas
                if line.count(',') == 2:
                    parts = line.split(',')
                    try:
                        # Try to parse the date
                        date_str = parts[0].strip()
                        if date_str:
                            datetime.strptime(date_str, '%Y-%m-%d')
                            current_date = date_str
                        elif current_date:  # Use previous date if current is empty
                            parts[0] = current_date
                        else:
                            continue  # Skip if no valid date available
                        
                        # Clean description
                        description = parts[1].strip()
                        description = ' '.join(description.split())  # Normalize spaces
                        description = description.replace('"', '')   # Remove quotes
                        
                        # Validate and clean amount
                        amount = float(parts[2].strip().replace(',', ''))
                        
                        # Reconstruct the line
                        cleaned_line = f"{parts[0].strip()},{description},{amount:.2f}"
                        cleaned_lines.append(cleaned_line)
                        
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping invalid line: {line} - Error: {str(e)}")
                        continue
        
        if not header_found:
            cleaned_lines.insert(0, 'date,description,amount')
        
        return '\n'.join(cleaned_lines)

    def process_single_file(self, pdf_path: str, output_path: Optional[str], password: Optional[str]) -> bool:
        """Process a single PDF file"""
        try:
            pdf_text = self.extract_text_from_pdf(pdf_path, password)
            if not pdf_text:
                self.failed_files.append((pdf_path, "Text extraction failed"))
                self.failure_count += 1
                return False

            prompt = self.create_parsing_prompt(pdf_text)
            response = self.query_ollama(prompt)
            
            if not response:
                self.failed_files.append((pdf_path, "LLM processing failed"))
                self.failure_count += 1
                return False

            csv_data = self.clean_csv_response(response)
            
            # Determine output path
            if not output_path:
                output_path = pdf_path.replace('.pdf', '_transactions.csv')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                csvfile.write(csv_data)
            
            self.success_count += 1
            logging.info(f"Successfully processed: {pdf_path}")
            return True
            
        except Exception as e:
            self.failed_files.append((pdf_path, str(e)))
            self.failure_count += 1
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return False

    def process_directory(self, 
                         dir_path: str, 
                         output_dir: str,
                         password_map: Optional[Dict[str, str]] = None,
                         default_password: Optional[str] = None,
                         recursive: bool = False) -> None:
        """Process all PDFs in a directory"""
        
        # Collect all PDF files
        pdf_files = []
        if recursive:
            for root, _, files in os.walk(dir_path):
                pdf_files.extend([
                    os.path.join(root, f) for f in files 
                    if f.lower().endswith('.pdf')
                ])
        else:
            pdf_files = [
                os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                if f.lower().endswith('.pdf')
            ]
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {dir_path}")
            return
        
        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files with progress bar
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            with ThreadPoolExecutor() as executor:
                futures = []
                for pdf_path in pdf_files:
                    filename = os.path.basename(pdf_path)
                    password = None
                    
                    # Determine password
                    if password_map and filename in password_map:
                        password = password_map[filename]
                    elif password_map and "default" in password_map:
                        password = password_map["default"]
                    else:
                        password = default_password
                    
                    # Determine output path
                    rel_path = os.path.relpath(pdf_path, dir_path)
                    output_path = os.path.join(
                        output_dir,
                        os.path.splitext(rel_path)[0] + '_transactions.csv'
                    )
                    
                    # Submit task
                    future = executor.submit(self.process_single_file, pdf_path, output_path, password)
                    futures.append(future)
                
                # Wait for completion and update progress
                for future in futures:
                    future.result()
                    pbar.update(1)

    def generate_report(self) -> str:
        """Generate processing summary report"""
        end_time = datetime.utcnow()
        processing_time = end_time - self.start_time
        
        report = [
            "\nProcessing Summary",
            "=" * 50,
            f"Processing completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"Total processing time: {processing_time}",
            f"Total files processed: {self.success_count + self.failure_count}",
            f"Successfully processed: {self.success_count}",
            f"Failed: {self.failure_count}",
            f"Success rate: {(self.success_count / (self.success_count + self.failure_count) * 100):.1f}%"
        ]
        
        if self.failed_files:
            report.extend([
                "\nFailed Files:",
                "-" * 50
            ])
            for file_path, error in self.failed_files:
                report.append(f"- {file_path}: {error}")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF bank statements to CSV using Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process single PDF:
    %(prog)s statement.pdf --password "mypassword"
    
  Process directory:
    %(prog)s --dir statements/ --password-file passwords.json --recursive
    
  Process with output directory:
    %(prog)s --dir statements/ --password "default123" --output-dir /path/to/output
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('pdf_path', nargs='?', help='Path to single PDF file')
    input_group.add_argument('--dir', help='Directory containing PDF files')
    
    # Processing options
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--password', help='Default password for encrypted PDFs')
    parser.add_argument('--password-file', help='JSON file mapping PDF names to passwords')
    parser.add_argument('--output-dir', help='Output directory for CSV files')
    parser.add_argument('--model', default='llama3:latest', help='Ollama model to use')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = BankStatementProcessor(args.model)
        
        # Handle password configuration
        password_map = None
        if args.password_file:
            try:
                with open(args.password_file, 'r') as f:
                    password_map = json.load(f)
            except Exception as e:
                logging.error(f"Error reading password file: {str(e)}")
                sys.exit(1)
        
        # Process files
        if args.dir:
            output_dir = args.output_dir or os.path.join(args.dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            processor.process_directory(
                args.dir,
                output_dir,
                password_map=password_map,
                default_password=args.password,
                recursive=args.recursive
            )
        else:
            # Single file processing
            output_path = None
            if args.output_dir:
                output_path = os.path.join(
                    args.output_dir,
                    os.path.basename(args.pdf_path).replace('.pdf', '_transactions.csv')
                )
            
            password = None
            if password_map:
                filename = os.path.basename(args.pdf_path)
                password = password_map.get(filename, password_map.get('default'))
            else:
                password = args.password
            
            success = processor.process_single_file(args.pdf_path, output_path, password)
            if not success:
                sys.exit(1)
        
        # Print report
        print(processor.generate_report())
        
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


