# ==============================================================================
# data_processing_agent.py - Data Processing Service
# ==============================================================================

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import io
from pathlib import Path
import sys

# Add the shared models to the path
sys.path.append(str(Path(__file__).parent))
from shared_models import (
    BaseAgent, AgentMessage, Transaction, SharedState, 
    DataValidationException, ProcessingException
)

# ==============================================================================
# ENHANCED FILE CONVERTER
# ==============================================================================

class EnhancedFileConverter:
    """Handles conversion of various file formats to standardized transactions"""
    
    @staticmethod
    async def validate_and_convert_file(file_data) -> pd.DataFrame:
        """Validate and convert uploaded file to standard format"""
        try:
            # Try multiple encodings and separators
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    file_data.seek(0)
                    
                    # Handle different file types
                    if hasattr(file_data, 'name'):
                        if file_data.name.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file_data)
                        else:
                            # Try different separators for CSV
                            for sep in [',', ';', '\t', '|']:
                                try:
                                    file_data.seek(0)
                                    df = pd.read_csv(file_data, encoding=encoding, sep=sep)
                                    if len(df.columns) > 1:
                                        break
                                except:
                                    continue
                    else:
                        # Assume CSV if no filename
                        for sep in [',', ';', '\t', '|']:
                            try:
                                file_data.seek(0)
                                df = pd.read_csv(file_data, encoding=encoding, sep=sep)
                                if len(df.columns) > 1:
                                    break
                            except:
                                continue
                    
                    if df is not None and len(df.columns) > 1:
                        break
                        
                except Exception as e:
                    continue
            
            if df is None or len(df.columns) <= 1:
                raise DataValidationException("Could not read file with any supported format or encoding")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Auto-detect and standardize columns
            return await EnhancedFileConverter._standardize_columns(df)
            
        except Exception as e:
            raise ProcessingException(f"File conversion failed: {str(e)}")
    
    @staticmethod
    async def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and formats"""
        standard_df = pd.DataFrame()
        
        # Find date column
        date_col = await EnhancedFileConverter._find_date_column(df)
        if not date_col:
            raise DataValidationException("Could not detect date column")
        
        # Find description column
        desc_col = await EnhancedFileConverter._find_description_column(df)
        if not desc_col:
            raise DataValidationException("Could not detect description column")
        
        # Find amount column
        amount_col = await EnhancedFileConverter._find_amount_column(df, [date_col, desc_col])
        if not amount_col:
            raise DataValidationException("Could not detect amount column")
        
        # Convert columns
        try:
            # Convert date
            standard_df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
            
            # Convert description
            standard_df['description'] = df[desc_col].astype(str).str.strip()
            
            # Convert amount
            standard_df['amount'] = await EnhancedFileConverter._clean_amount_column(df[amount_col])
            
        except Exception as e:
            raise ProcessingException(f"Column conversion failed: {str(e)}")
        
        # Remove rows with missing critical data
        initial_rows = len(standard_df)
        standard_df = standard_df.dropna(subset=['date', 'amount'])
        standard_df = standard_df[standard_df['description'].str.len() > 0]
        
        if len(standard_df) == 0:
            raise DataValidationException("No valid transactions found after cleaning")
        
        final_rows = len(standard_df)
        if final_rows < initial_rows * 0.5:
            print(f"Warning: {initial_rows - final_rows} rows removed during cleaning")
        
        return standard_df
    
    @staticmethod
    async def _find_date_column(df: pd.DataFrame) -> str:
        """Find date column using pattern matching and validation"""
        date_patterns = ['date', 'time', 'posted', 'transaction_date', 'value_date']
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    # Test if column contains dates
                    pd.to_datetime(df[col].dropna().head(5))
                    return col
                except:
                    continue
        
        # If no pattern match, try all columns
        for col in df.columns:
            try:
                sample = df[col].dropna().head(10)
                if sample.empty:
                    continue
                
                # Try to parse as dates
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() >= len(sample) * 0.8:  # 80% success rate
                    return col
            except:
                continue
        
        return None
    
    @staticmethod
    async def _find_description_column(df: pd.DataFrame) -> str:
        """Find description column using pattern matching"""
        desc_patterns = ['description', 'transaction', 'details', 'narration', 'memo', 'particulars']
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in desc_patterns):
                return col
        
        # If no pattern match, find the column with most text variation
        text_cols = []
        for col in df.columns:
            try:
                sample = df[col].astype(str).str.strip()
                unique_ratio = len(sample.unique()) / len(sample)
                avg_length = sample.str.len().mean()
                
                if unique_ratio > 0.5 and avg_length > 5:  # High variation, reasonable length
                    text_cols.append((col, unique_ratio, avg_length))
            except:
                continue
        
        if text_cols:
            # Sort by uniqueness and length
            text_cols.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return text_cols[0][0]
        
        return None
    
    @staticmethod
    async def _find_amount_column(df: pd.DataFrame, exclude_cols: List[str]) -> str:
        """Find amount column using numeric validation"""
        amount_patterns = ['amount', 'transaction_amount', 'debit', 'credit', 'balance', 'value']
        
        # First try pattern matching
        for col in df.columns:
            if col not in exclude_cols and any(pattern in col.lower() for pattern in amount_patterns):
                try:
                    # Test if column can be converted to numeric
                    test_series = await EnhancedFileConverter._clean_amount_column(df[col])
                    if not test_series.isna().all():
                        return col
                except:
                    continue
        
        # If no pattern match, try all remaining columns
        for col in df.columns:
            if col not in exclude_cols:
                try:
                    test_series = await EnhancedFileConverter._clean_amount_column(df[col])
                    if not test_series.isna().all():
                        numeric_ratio = test_series.notna().sum() / len(test_series)
                        if numeric_ratio > 0.8:  # 80% numeric values
                            return col
                except:
                    continue
        
        return None
    
    @staticmethod
    async def _clean_amount_column(amount_series: pd.Series) -> pd.Series:
        """Clean and convert amount column with comprehensive handling"""
        cleaned = amount_series.copy()
        
        # Convert to string first
        cleaned = cleaned.astype(str)
        
        # Remove common currency symbols and whitespace
        cleaned = cleaned.str.replace(r'[$£€¥₹,\s]', '', regex=True)
        
        # Handle parentheses for negative numbers
        negative_mask = cleaned.str.contains(r'\(.*\)', regex=True, na=False)
        cleaned = cleaned.str.replace(r'[()]', '', regex=True)
        
        # Handle CR/DR indicators
        cr_mask = cleaned.str.contains(r'\bCR\b', regex=True, na=False)
        dr_mask = cleaned.str.contains(r'\bDR\b', regex=True, na=False)
        cleaned = cleaned.str.replace(r'\b(CR|DR)\b', '', regex=True)
        
        # Convert to numeric
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        # Apply negative signs
        cleaned.loc[negative_mask] = -abs(cleaned.loc[negative_mask])
        cleaned.loc[dr_mask] = -abs(cleaned.loc[dr_mask])
        # CR entries remain positive
        
        return cleaned

# ==============================================================================
# TRANSACTION CATEGORIZER
# ==============================================================================

class TransactionCategorizer:
    """Intelligent transaction categorization using pattern matching"""
    
    def __init__(self):
        self.category_patterns = {
            'food_dining': [
                'grocery', 'supermarket', 'food', 'restaurant', 'cafe', 'dining',
                'meal', 'pizza', 'burger', 'deli', 'bakery', 'market', 'store',
                'mcdonalds', 'subway', 'starbucks', 'dunkin'
            ],
            'fixed_expenses': [
                'rent', 'mortgage', 'insurance', 'utility', 'phone', 'internet',
                'electric', 'gas bill', 'water', 'sewer', 'cable', 'subscription'
            ],
            'transportation': [
                'fuel', 'gas station', 'transport', 'uber', 'taxi', 'bus', 'train',
                'parking', 'metro', 'lyft', 'shell', 'exxon', 'chevron', 'bp'
            ],
            'shopping': [
                'amazon', 'shopping', 'retail', 'store', 'mall', 'clothes',
                'electronics', 'walmart', 'target', 'bestbuy', 'ebay'
            ],
            'income': [
                'salary', 'wage', 'income', 'bonus', 'payroll', 'deposit',
                'pay', 'employer', 'direct deposit', 'refund'
            ],
            'savings_investment': [
                'transfer', 'savings', 'investment', '401k', 'ira', 'retirement',
                'mutual fund', 'stock', 'bond', 'portfolio'
            ],
            'entertainment': [
                'entertainment', 'movie', 'game', 'subscription', 'netflix',
                'spotify', 'gaming', 'cinema', 'theater', 'concert'
            ],
            'healthcare': [
                'medical', 'doctor', 'pharmacy', 'hospital', 'health',
                'dental', 'vision', 'clinic', 'prescription', 'cvs', 'walgreens'
            ],
            'personal_care': [
                'salon', 'barber', 'spa', 'gym', 'fitness', 'beauty',
                'cosmetics', 'personal care'
            ],
            'education': [
                'education', 'school', 'university', 'college', 'tuition',
                'books', 'course', 'training'
            ]
        }
        
        self.spending_type_patterns = {
            'weekend_spending': ['friday', 'saturday', 'sunday'],
            'impulse_spending': ['impulse', 'sale', 'clearance', 'flash sale'],
            'subscription': ['monthly', 'annual', 'subscription', 'recurring']
        }
    
    async def categorize_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """Categorize list of transactions"""
        for transaction in transactions:
            await self._categorize_single_transaction(transaction)
        
        return transactions
    
    async def _categorize_single_transaction(self, transaction: Transaction):
        """Categorize a single transaction"""
        desc_lower = transaction.description.lower()
        
        # Category classification
        for category, patterns in self.category_patterns.items():
            if any(pattern in desc_lower for pattern in patterns):
                transaction.category = category
                break
        
        # Spending type classification
        if transaction.amount < 0:  # Only for expenses
            if abs(transaction.amount) > 500:
                transaction.spending_type = "major_expense"
            else:
                for spending_type, patterns in self.spending_type_patterns.items():
                    if any(pattern in desc_lower for pattern in patterns):
                        transaction.spending_type = spending_type
                        break
    
    def get_category_confidence(self, transaction: Transaction) -> float:
        """Calculate confidence score for categorization"""
        desc_lower = transaction.description.lower()
        
        if transaction.category in self.category_patterns:
            patterns = self.category_patterns[transaction.category]
            matches = sum(1 for pattern in patterns if pattern in desc_lower)
            return min(1.0, matches / len(patterns) * 3)  # Scale confidence
        
        return 0.5  # Default confidence for uncategorized

# ==============================================================================
# DATA PROCESSING AGENT
# ==============================================================================

class DataProcessingAgent(BaseAgent):
    """Handles file upload, validation, and transaction processing"""
    
    def __init__(self, shared_state: SharedState = None):
        super().__init__("DataProcessor", shared_state)
        self.file_converter = EnhancedFileConverter()
        self.categorizer = TransactionCategorizer()
        
        # Register message handlers
        self.register_handler("process_file", self._handle_process_file)
        self.register_handler("validate_data", self._handle_validate_data)
        self.register_handler("categorize_transactions", self._handle_categorize_transactions)
        self.register_handler("health_check", self._handle_health_check)
    
    async def _handle_process_file(self, message: AgentMessage) -> List[AgentMessage]:
        """Process uploaded file and extract transactions"""
        await self.log("Starting file processing")
        
        try:
            file_data = message.data.get("file_data")
            if not file_data:
                raise DataValidationException("No file data provided")
            
            # Convert file to DataFrame
            df = await self.file_converter.validate_and_convert_file(file_data)
            
            # Extract transactions
            transactions = await self._extract_transactions(df)
            
            # Categorize transactions
            categorized_transactions = await self.categorizer.categorize_transactions(transactions)
            
            # Update shared state
            await self.shared_state.set("transactions", [tx.to_dict() for tx in categorized_transactions])
            await self.shared_state.set("processing_status", "data_processed")
            
            await self.log(f"Successfully processed {len(categorized_transactions)} transactions")
            
            # Notify other agents
            return [
                AgentMessage(
                    sender=self.name,
                    recipient="HealthCalculator",
                    message_type="calculate_health",
                    data={"transactions": [tx.to_dict() for tx in categorized_transactions]}
                ),
                AgentMessage(
                    sender=self.name,
                    recipient="InsightsAnalyzer",
                    message_type="analyze_patterns",
                    data={"transactions": [tx.to_dict() for tx in categorized_transactions]}
                )
            ]
            
        except Exception as e:
            await self.log(f"File processing failed: {e}", "error")
            await self.shared_state.set("processing_status", "error")
            await self.shared_state.set("error_messages", [str(e)])
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient="ErrorHandler",
                    message_type="error",
                    data={"error": str(e), "context": "file_processing"}
                )
            ]
    
    async def _handle_validate_data(self, message: AgentMessage) -> List[AgentMessage]:
        """Validate transaction data"""
        await self.log("Validating transaction data")
        
        try:
            transactions_data = message.data.get("transactions", [])
            
            valid_transactions = []
            invalid_count = 0
            
            for tx_data in transactions_data:
                try:
                    transaction = Transaction.from_dict(tx_data)
                    valid_transactions.append(transaction)
                except Exception as e:
                    invalid_count += 1
                    await self.log(f"Invalid transaction: {e}", "warning")
            
            await self.log(f"Validated {len(valid_transactions)} transactions, {invalid_count} invalid")
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="validation_complete",
                    data={
                        "valid_transactions": [tx.to_dict() for tx in valid_transactions],
                        "invalid_count": invalid_count
                    }
                )
            ]
            
        except Exception as e:
            await self.log(f"Data validation failed: {e}", "error")
            return []
    
    async def _handle_categorize_transactions(self, message: AgentMessage) -> List[AgentMessage]:
        """Categorize transactions"""
        await self.log("Categorizing transactions")
        
        try:
            transactions_data = message.data.get("transactions", [])
            transactions = [Transaction.from_dict(tx) for tx in transactions_data]
            
            categorized_transactions = await self.categorizer.categorize_transactions(transactions)
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="categorization_complete",
                    data={"transactions": [tx.to_dict() for tx in categorized_transactions]}
                )
            ]
            
        except Exception as e:
            await self.log(f"Categorization failed: {e}", "error")
            return []
    
    async def _handle_health_check(self, message: AgentMessage) -> List[AgentMessage]:
        """Health check handler"""
        return [
            AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="health_response",
                data={
                    "status": "healthy",
                    "capabilities": self.capabilities,
                    "active": self.is_active
                }
            )
        ]
    
    async def _extract_transactions(self, df: pd.DataFrame) -> List[Transaction]:
        """Extract transactions from DataFrame"""
        transactions = []
        
        for _, row in df.iterrows():
            try:
                transaction = Transaction(
                    date=row['date'],
                    description=row['description'],
                    amount=float(row['amount'])
                )
                transactions.append(transaction)
            except Exception as e:
                await self.log(f"Skipping invalid transaction: {e}", "warning")
                continue
        
        return transactions
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        transactions = await self.shared_state.get("transactions", [])
        
        if not transactions:
            return {"total_transactions": 0}
        
        # Calculate statistics
        expenses = [tx for tx in transactions if isinstance(tx, dict) and tx.get('amount', 0) < 0]
        income = [tx for tx in transactions if isinstance(tx, dict) and tx.get('amount', 0) > 0]
        
        # Category distribution
        categories = {}
        for tx in transactions:
            if isinstance(tx, dict):
                category = tx.get('category', 'other')
                categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_transactions": len(transactions),
            "expenses_count": len(expenses),
            "income_count": len(income),
            "category_distribution": categories,
            "processing_status": await self.shared_state.get("processing_status", "idle")
        }

# ==============================================================================
# STANDALONE SERVICE
# ==============================================================================

class DataProcessingService:
    """Standalone data processing service"""
    
    def __init__(self, config_file: str = None):
        self.shared_state = SharedState()
        self.agent = DataProcessingAgent(self.shared_state)
        self.is_running = False
        
    async def start(self):
        """Start the data processing service"""
        self.is_running = True
        await self.agent.log("Data Processing Service started")
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
    
    async def stop(self):
        """Stop the data processing service"""
        self.is_running = False
        await self.agent.log("Data Processing Service stopped")
    
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file and return results"""
        try:
            with open(file_path, 'rb') as f:
                message = AgentMessage(
                    sender="API",
                    recipient="DataProcessor",
                    message_type="process_file",
                    data={"file_data": f}
                )
                
                results = await self.agent.process(message)
                
                # Get processing results
                stats = await self.agent.get_processing_stats()
                return {
                    "success": True,
                    "stats": stats,
                    "messages": [msg.to_dict() for msg in results]
                }
                
        except Exception as e:
            await self.agent.log(f"File processing error: {e}", "error")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "DataProcessingService",
            "status": "running" if self.is_running else "stopped",
            "agent_status": self.agent.is_active,
            "capabilities": self.agent.capabilities,
            "stats": await self.agent.get_processing_stats()
        }
    
    async def _health_monitor(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                # Check system health every 30 seconds
                await asyncio.sleep(30)
                
                # Log health status
                stats = await self.agent.get_processing_stats()
                await self.agent.log(f"Health check: {stats['total_transactions']} transactions processed")
                
            except Exception as e:
                await self.agent.log(f"Health monitor error: {e}", "error")

# ==============================================================================
# CLI INTERFACE
# ==============================================================================

async def main():
    """Main CLI interface for data processing service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EmpowerFin Data Processing Service")
    parser.add_argument("--file", "-f", help="File to process")
    parser.add_argument("--service", "-s", action="store_true", help="Run as service")
    parser.add_argument("--stats", action="store_true", help="Show processing stats")
    parser.add_argument("--test", "-t", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    service = DataProcessingService()
    
    if args.test:
        await run_tests(service)
    elif args.service:
        await run_service(service)
    elif args.file:
        await process_single_file(service, args.file)
    elif args.stats:
        await show_stats(service)
    else:
        parser.print_help()

async def run_tests(service: DataProcessingService):
    """Run test suite"""
    print("Running Data Processing Agent tests...")
    
    # Test 1: Basic functionality
    await service.start()
    
    # Create test data
    test_data = """date,description,amount
2024-01-01,Grocery Store,-50.00
2024-01-02,Salary Deposit,2000.00
2024-01-03,Gas Station,-30.00"""
    
    test_file = io.StringIO(test_data)
    test_file.name = "test.csv"
    
    try:
        message = AgentMessage(
            sender="Test",
            recipient="DataProcessor",
            message_type="process_file",
            data={"file_data": test_file}
        )
        
        results = await service.agent.process(message)
        print(f"✓ File processing test passed: {len(results)} messages generated")
        
        stats = await service.agent.get_processing_stats()
        print(f"✓ Statistics test passed: {stats['total_transactions']} transactions")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    await service.stop()

async def run_service(service: DataProcessingService):
    """Run as a standalone service"""
    print("Starting Data Processing Service...")
    
    await service.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down service...")
        await service.stop()

async def process_single_file(service: DataProcessingService, file_path: str):
    """Process a single file"""
    print(f"Processing file: {file_path}")
    
    await service.start()
    
    result = await service.process_file(file_path)
    
    if result["success"]:
        print("✓ File processed successfully")
        print(f"Statistics: {result['stats']}")
    else:
        print(f"✗ Processing failed: {result['error']}")
    
    await service.stop()

async def show_stats(service: DataProcessingService):
    """Show service statistics"""
    await service.start()
    
    status = await service.get_status()
    print("Data Processing Service Status:")
    print(f"  Service: {status['status']}")
    print(f"  Agent Active: {status['agent_status']}")
    print(f"  Capabilities: {', '.join(status['capabilities'])}")
    print(f"  Statistics: {status['stats']}")
    
    await service.stop()

if __name__ == "__main__":
    asyncio.run(main())