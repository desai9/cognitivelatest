# ==============================================================================
# file_processor_agent.py - LLM-Enhanced File Processing Agent
# ==============================================================================

import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage

from config_manager import llm_manager, app_config, session_manager
from data_models import EnhancedTransaction, LLMAnalysisResult


class FileProcessorAgent:
    """Intelligent file processing with LLM-enhanced column detection"""
    
    def __init__(self):
        self.llm = llm_manager.get_client()
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.txt']
        
    def process_uploaded_file(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], LLMAnalysisResult]:
        """Main entry point for file processing"""
        try:
            # Step 1: Basic file reading
            df = self._read_file_with_format_detection(uploaded_file)
            
            if df is None:
                return None, LLMAnalysisResult(
                    success=False,
                    error_message="Could not read file - unsupported format or corrupted file"
                )
            
            # Step 2: Intelligent column detection
            if self.llm:
                # Use LLM for smart column mapping
                processed_df, analysis_result = self._llm_enhanced_column_detection(df)
            else:
                # Fallback to rule-based detection
                processed_df, analysis_result = self._rule_based_column_detection(df)
            
            # Step 3: Data validation and cleaning
            if processed_df is not None:
                processed_df = self._clean_and_validate_data(processed_df)
            
            return processed_df, analysis_result
            
        except Exception as e:
            return None, LLMAnalysisResult(
                success=False,
                error_message=f"File processing failed: {str(e)}"
            )
    
    def _read_file_with_format_detection(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Read file with intelligent format detection"""
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        try:
            # Excel files
            if file_extension in ['xlsx', 'xls']:
                return pd.read_excel(uploaded_file)
            
            # CSV and text files - try different separators and encodings
            elif file_extension in ['csv', 'txt']:
                return self._read_csv_with_detection(uploaded_file)
            
            else:
                # Try to read as CSV anyway
                return self._read_csv_with_detection(uploaded_file)
                
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None
    
    def _read_csv_with_detection(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Intelligent CSV reading with separator and encoding detection"""
        separators = [',', ';', '\t', '|', ' ']
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
                    
                    # Validate that we have multiple columns and reasonable data
                    if len(df.columns) > 1 and len(df) > 0:
                        return df
                        
                except Exception:
                    continue
        
        # Last resort - try with python engine and error handling
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            return None
    
    def _llm_enhanced_column_detection(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], LLMAnalysisResult]:
        """Use LLM to intelligently detect and map columns"""
        try:
            # Prepare sample data for LLM analysis
            sample_data = self._prepare_sample_for_llm(df)
            
            # Create LLM prompt for column detection
            prompt = self._create_column_detection_prompt(sample_data)
            
            # Get LLM response
            messages = [
                SystemMessage(content="You are a financial data analysis expert. Analyze bank statement data and identify column mappings accurately. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse and apply LLM response
            return self._parse_and_apply_llm_mapping(df, response.content)
            
        except Exception as e:
            # Fallback to rule-based detection
            return self._rule_based_column_detection(df)
    
    def _prepare_sample_for_llm(self, df: pd.DataFrame) -> Dict:
        """Prepare sample data for LLM analysis"""
        # Get first few rows for analysis
        sample_size = min(5, len(df))
        sample_df = df.head(sample_size)
        
        return {
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'row_count': len(df),
            'sample_rows': sample_df.values.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'column_samples': {col: df[col].dropna().head(3).tolist() for col in df.columns}
        }
    
    def _create_column_detection_prompt(self, sample_data: Dict) -> str:
        """Create comprehensive prompt for column detection"""
        return f"""
        Analyze this bank statement data and identify the correct columns for financial transactions.
        
        File Information:
        - Number of columns: {sample_data['column_count']}
        - Number of rows: {sample_data['row_count']}
        - Column names: {sample_data['columns']}
        
        Sample data for each column:
        {json.dumps(sample_data['column_samples'], indent=2)}
        
        Data types detected:
        {json.dumps(sample_data['data_types'], indent=2)}
        
        I need you to identify which columns represent:
        1. DATE - Transaction date (required)
        2. DESCRIPTION - Transaction description/memo/details (required)  
        3. AMOUNT - Transaction amount in currency (required)
           - May be positive/negative or have separate debit/credit columns
           - Could contain currency symbols, commas, parentheses for negatives
        
        Additional columns that might exist:
        4. BALANCE - Account balance after transaction
        5. REFERENCE - Transaction reference/ID
        6. CATEGORY - Pre-categorized transaction type
        
        Instructions:
        - Look for date patterns (YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, etc.)
        - Description usually contains merchant names, transaction details
        - Amount columns contain numeric values, may have currency symbols
        - If there are separate debit/credit columns, specify both
        - Be confident in your assessment based on the sample data
        
        Respond in this exact JSON format:
        {{
            "date_column": "exact_column_name",
            "description_column": "exact_column_name",
            "amount_column": "exact_column_name",
            "debit_column": "exact_column_name_or_null",
            "credit_column": "exact_column_name_or_null", 
            "balance_column": "exact_column_name_or_null",
            "reference_column": "exact_column_name_or_null",
            "confidence": "high|medium|low",
            "reasoning": "brief explanation of your analysis",
            "notes": "any special handling needed for amount column"
        }}
        """
    
    def _parse_and_apply_llm_mapping(self, df: pd.DataFrame, llm_response: str) -> Tuple[Optional[pd.DataFrame], LLMAnalysisResult]:
        """Parse LLM response and apply column mapping"""
        try:
            # Clean the LLM response to extract JSON
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = llm_response
            
            # Parse JSON response
            mapping = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['date_column', 'description_column', 'amount_column']
            for field in required_fields:
                if not mapping.get(field) or mapping[field] not in df.columns:
                    raise ValueError(f"Invalid or missing {field}")
            
            # Apply mapping to create standardized DataFrame
            result_df = self._create_standardized_dataframe(df, mapping)
            
            # Create success result
            analysis_result = LLMAnalysisResult(
                success=True,
                data=mapping,
                confidence=mapping.get('confidence', 'medium'),
                reasoning=mapping.get('reasoning', 'LLM successfully mapped columns')
            )
            
            # Store analysis for user feedback
            session_manager.save_analysis_result('column_detection', mapping)
            
            return result_df, analysis_result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to rule-based detection
            return self._rule_based_column_detection(df)
    
    def _create_standardized_dataframe(self, df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
        """Create standardized DataFrame from column mapping"""
        result_df = pd.DataFrame()
        
        # Date column
        date_col = mapping['date_column']
        result_df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
        
        # Description column
        desc_col = mapping['description_column']
        result_df['description'] = df[desc_col].astype(str).str.strip()
        
        # Amount column(s) - handle single amount or separate debit/credit
        if mapping.get('debit_column') and mapping.get('credit_column'):
            # Separate debit/credit columns
            debit_col = mapping['debit_column']
            credit_col = mapping['credit_column']
            
            result_df['amount'] = self._combine_debit_credit_columns(df, debit_col, credit_col)
        else:
            # Single amount column
            amount_col = mapping['amount_column']
            result_df['amount'] = self._clean_amount_column(df[amount_col])
        
        # Optional columns
        if mapping.get('balance_column') and mapping['balance_column'] in df.columns:
            result_df['balance'] = self._clean_amount_column(df[mapping['balance_column']])
        
        if mapping.get('reference_column') and mapping['reference_column'] in df.columns:
            result_df['reference'] = df[mapping['reference_column']].astype(str)
        
        return result_df.dropna(subset=['date', 'description', 'amount'])
    
    def _combine_debit_credit_columns(self, df: pd.DataFrame, debit_col: str, credit_col: str) -> pd.Series:
        """Combine separate debit and credit columns into single amount"""
        debit_series = self._clean_amount_column(df[debit_col]).fillna(0)
        credit_series = self._clean_amount_column(df[credit_col]).fillna(0)
        
        # Debits are negative, credits are positive
        return credit_series - debit_series
    
    def _clean_amount_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert amount column to numeric"""
        # Convert to string first
        clean_series = series.astype(str)
        
        # Remove currency symbols, commas, and whitespace
        clean_series = clean_series.str.replace(r'[$£€¥₹,\s]', '', regex=True)
        
        # Handle negative numbers in parentheses (e.g., "(100.00)" -> "-100.00")
        negative_mask = clean_series.str.contains(r'\(.*\)', na=False)
        clean_series = clean_series.str.replace(r'[()]', '', regex=True)
        
        # Convert to numeric
        numeric_series = pd.to_numeric(clean_series, errors='coerce')
        
        # Apply negative sign where parentheses were found
        numeric_series.loc[negative_mask] = -abs(numeric_series.loc[negative_mask])
        
        return numeric_series
    
    def _rule_based_column_detection(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], LLMAnalysisResult]:
        """Fallback rule-based column detection"""
        try:
            # Clean column names for easier matching
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Find columns using keyword matching
            date_col = self._find_column_by_keywords(df, ['date', 'time', 'posted', 'transaction_date'])
            desc_col = self._find_column_by_keywords(df, ['description', 'transaction', 'details', 'memo', 'payee'])
            amount_col = self._find_column_by_keywords(df, ['amount', 'value', 'sum', 'total'])
            
            # Check for separate debit/credit columns
            debit_col = self._find_column_by_keywords(df, ['debit', 'withdrawal', 'out'])
            credit_col = self._find_column_by_keywords(df, ['credit', 'deposit', 'in'])
            
            # Validate required columns
            if not date_col or not desc_col:
                raise ValueError("Could not identify required date and description columns")
            
            if not amount_col and not (debit_col and credit_col):
                raise ValueError("Could not identify amount column(s)")
            
            # Create mapping
            mapping = {
                'date_column': date_col,
                'description_column': desc_col,
                'amount_column': amount_col,
                'debit_column': debit_col,
                'credit_column': credit_col,
                'confidence': 'medium',
                'reasoning': 'Rule-based detection using keyword matching'
            }
            
            # Create standardized DataFrame
            result_df = self._create_standardized_dataframe(df, mapping)
            
            analysis_result = LLMAnalysisResult(
                success=True,
                data=mapping,
                confidence='medium',
                reasoning='Successfully applied rule-based column detection'
            )
            
            return result_df, analysis_result
            
        except Exception as e:
            return None, LLMAnalysisResult(
                success=False,
                error_message=f"Rule-based detection failed: {str(e)}"
            )
    
    def _find_column_by_keywords(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find column name that matches any of the given keywords"""
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                return col
        return None
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data cleaning and validation"""
        # Remove rows with missing critical data
        df = df.dropna(subset=['date', 'description', 'amount'])
        
        # Remove rows with zero amounts (usually just informational)
        df = df[df['amount'] != 0]
        
        # Sort by date
        df = df.sort_values('date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Additional validation
        if len(df) == 0:
            raise ValueError("No valid transactions found after cleaning")
        
        return df
    
    def convert_to_enhanced_transactions(self, df: pd.DataFrame) -> List[EnhancedTransaction]:
        """Convert DataFrame to list of EnhancedTransaction objects"""
        transactions = []
        
        for _, row in df.iterrows():
            transaction = EnhancedTransaction(
                date=row['date'],
                description=row['description'],
                amount=float(row['amount'])
            )
            
            # Add optional fields if present
            if 'balance' in row and pd.notna(row['balance']):
                transaction.balance = float(row['balance'])
            
            if 'reference' in row and pd.notna(row['reference']):
                transaction.reference = str(row['reference'])
            
            transactions.append(transaction)
        
        return transactions
    
    def get_file_analysis_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of file analysis"""
        if df is None or df.empty:
            return {'status': 'failed', 'message': 'No data processed'}
        
        # Basic statistics
        total_transactions = len(df)
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_flow = total_income - total_expenses
        
        return {
            'status': 'success',
            'total_transactions': total_transactions,
            'date_range': date_range,
            'financial_summary': {
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_flow': net_flow
            },
            'data_quality': {
                'missing_dates': df['date'].isna().sum(),
                'missing_descriptions': df['description'].isna().sum(),
                'missing_amounts': df['amount'].isna().sum(),
                'zero_amounts': (df['amount'] == 0).sum()
            }
        }


class BatchFileProcessor:
    """Process multiple files in batch"""
    
    def __init__(self):
        self.processor = FileProcessorAgent()
        self.processed_files = []
    
    def process_multiple_files(self, uploaded_files: List) -> Dict:
        """Process multiple uploaded files"""
        results = {
            'successful': [],
            'failed': [],
            'combined_transactions': [],
            'summary': {}
        }
        
        for uploaded_file in uploaded_files:
            try:
                df, analysis_result = self.processor.process_uploaded_file(uploaded_file)
                
                if analysis_result.success and df is not None:
                    transactions = self.processor.convert_to_enhanced_transactions(df)
                    
                    results['successful'].append({
                        'filename': uploaded_file.name,
                        'transactions': len(transactions),
                        'analysis': analysis_result.to_dict()
                    })
                    
                    results['combined_transactions'].extend(transactions)
                else:
                    results['failed'].append({
                        'filename': uploaded_file.name,
                        'error': analysis_result.error_message
                    })
                    
            except Exception as e:
                results['failed'].append({
                    'filename': uploaded_file.name,
                    'error': str(e)
                })
        
        # Generate combined summary
        if results['combined_transactions']:
            results['summary'] = self._generate_batch_summary(results['combined_transactions'])
        
        return results
    
    def _generate_batch_summary(self, transactions: List[EnhancedTransaction]) -> Dict:
        """Generate summary for batch processed transactions"""
        if not transactions:
            return {}
        
        total_income = sum(tx.amount for tx in transactions if tx.amount > 0)
        total_expenses = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
        
        return {
            'total_files_processed': len(self.processed_files),
            'total_transactions': len(transactions),
            'date_range': f"{min(tx.date for tx in transactions)} to {max(tx.date for tx in transactions)}",
            'financial_overview': {
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_flow': total_income - total_expenses
            }
        }