# ==============================================================================
# categorization_agent.py - LLM-Powered Transaction Categorization Agent
# ==============================================================================

import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage

from config_manager import llm_manager, app_config
from data_models import EnhancedTransaction, LLMAnalysisResult


class TransactionCategorizationAgent:
    """Intelligent transaction categorization with LLM enhancement"""
    
    def __init__(self):
        self.llm = llm_manager.get_client()
        self.category_schema = app_config.category_schema
        self.category_keywords = app_config.category_keywords
        self.batch_size = 10  # Process transactions in batches for efficiency
        
    def categorize_transactions(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """Main entry point for transaction categorization"""
        if not transactions:
            return [], LLMAnalysisResult(success=False, error_message="No transactions to categorize")
        
        try:
            if self.llm:
                # Use LLM for intelligent categorization
                return self._llm_enhanced_categorization(transactions)
            else:
                # Fallback to rule-based categorization
                return self._rule_based_categorization(transactions)
                
        except Exception as e:
            st.warning(f"Categorization failed, using basic fallback: {e}")
            return self._basic_categorization(transactions)
    
    def _llm_enhanced_categorization(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """Use LLM for intelligent transaction categorization"""
        categorized_transactions = []
        analysis_results = []
        
        # Process transactions in batches
        for i in range(0, len(transactions), self.batch_size):
            batch = transactions[i:i + self.batch_size]
            
            try:
                categorized_batch, batch_analysis = self._categorize_batch_with_llm(batch)
                categorized_transactions.extend(categorized_batch)
                analysis_results.append(batch_analysis)
                
            except Exception as e:
                # Fallback to rule-based for this batch
                fallback_batch = self._rule_based_categorization_batch(batch)
                categorized_transactions.extend(fallback_batch)
        
        # Combine analysis results
        overall_analysis = LLMAnalysisResult(
            success=True,
            data={
                'total_transactions': len(transactions),
                'llm_processed': sum(1 for result in analysis_results if result.success),
                'fallback_processed': len(transactions) - sum(1 for result in analysis_results if result.success),
                'confidence_distribution': self._calculate_confidence_distribution(categorized_transactions)
            },
            confidence='high' if all(result.success for result in analysis_results) else 'medium',
            reasoning='LLM-enhanced categorization with intelligent batch processing'
        )
        
        return categorized_transactions, overall_analysis
    
    def _categorize_batch_with_llm(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """Categorize a batch of transactions using LLM"""
        # Prepare transaction data for LLM
        transaction_data = []
        for i, tx in enumerate(transactions):
            transaction_data.append({
                'id': i,
                'description': tx.description,
                'amount': tx.amount,
                'date': tx.date
            })
        
        # Create comprehensive categorization prompt
        prompt = self._create_categorization_prompt(transaction_data)
        
        # Get LLM response
        messages = [
            SystemMessage(content="You are a financial transaction categorization expert. Analyze transactions and categorize them accurately based on description, amount, and context. Always respond with valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse and apply categorizations
        return self._parse_and_apply_categorizations(transactions, response.content)
    
    def _create_categorization_prompt(self, transaction_data: List[Dict]) -> str:
        """Create comprehensive prompt for transaction categorization"""
        return f"""
        Categorize these financial transactions using the provided category system.
        
        CATEGORY SYSTEM:
        {json.dumps(self.category_schema, indent=2)}
        
        TRANSACTIONS TO CATEGORIZE:
        {json.dumps(transaction_data, indent=2)}
        
        For each transaction, analyze:
        1. Transaction description to identify merchant/purpose
        2. Amount to understand transaction significance
        3. Date for potential seasonal patterns
        
        Determine for each transaction:
        - Main category (from the category system above)
        - Subcategory (from the available subcategories)
        - Spending behavior: regular, impulse, seasonal, major_purchase, recurring
        - Confidence level: high, medium, low
        - Brief reasoning for the categorization
        
        CATEGORIZATION GUIDELINES:
        - Regular transactions: predictable, recurring expenses
        - Impulse: unplanned, discretionary purchases
        - Seasonal: holiday, vacation, weather-related spending
        - Major purchase: large, significant one-time expenses
        - Recurring: subscriptions, monthly services
        
        Income transactions (positive amounts):
        - Salary/wages should be 'income' -> 'salary'
        - Investment returns should be 'income' -> 'investment_returns'
        - Business income should be 'income' -> 'business_income'
        
        Respond in this exact JSON format:
        {{
            "categorizations": [
                {{
                    "id": 0,
                    "category": "category_name",
                    "subcategory": "subcategory_name",
                    "spending_behavior": "behavior_type",
                    "confidence": "high|medium|low",
                    "reasoning": "brief explanation (max 50 words)"
                }}
            ],
            "batch_confidence": "high|medium|low",
            "notes": "any patterns or insights observed"
        }}
        """
    
    def _parse_and_apply_categorizations(self, transactions: List[EnhancedTransaction], llm_response: str) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """Parse LLM response and apply categorizations"""
        try:
            # Clean and parse JSON response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = llm_response
            
            result = json.loads(json_str)
            categorizations = result.get('categorizations', [])
            
            # Apply categorizations to transactions
            for tx in transactions:
                # Find matching categorization by index
                matching_cat = None
                for cat in categorizations:
                    if cat.get('id') < len(transactions):
                        matching_cat = cat
                        break
                
                if matching_cat:
                    tx.category = matching_cat.get('category', 'other')
                    tx.subcategory = matching_cat.get('subcategory', 'uncategorized')
                    tx.spending_type = matching_cat.get('spending_behavior', 'regular')
                    tx.confidence = self._convert_confidence_to_numeric(matching_cat.get('confidence', 'medium'))
                    tx.llm_reasoning = matching_cat.get('reasoning', '')
                else:
                    # Fallback categorization
                    self._apply_fallback_categorization(tx)
            
            analysis_result = LLMAnalysisResult(
                success=True,
                data=result,
                confidence=result.get('batch_confidence', 'medium'),
                reasoning=f"LLM categorized {len(categorizations)} transactions"
            )
            
            return transactions, analysis_result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to rule-based for this batch
            fallback_transactions = self._rule_based_categorization_batch(transactions)
            
            analysis_result = LLMAnalysisResult(
                success=False,
                error_message=f"LLM parsing failed: {e}, used fallback",
                confidence='low'
            )
            
            return fallback_transactions, analysis_result
    
    def _convert_confidence_to_numeric(self, confidence_str: str) -> float:
        """Convert confidence string to numeric value"""
        confidence_map = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        return confidence_map.get(confidence_str.lower(), 0.7)
    
    def _rule_based_categorization(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """Comprehensive rule-based categorization fallback"""
        categorized_transactions = []
        
        for tx in transactions:
            self._apply_rule_based_categorization(tx)
            categorized_transactions.append(tx)
        
        analysis_result = LLMAnalysisResult(
            success=True,
            data={'method': 'rule_based', 'total_processed': len(transactions)},
            confidence='medium',
            reasoning='Applied rule-based categorization using keyword matching'
        )
        
        return categorized_transactions, analysis_result
    
    def _rule_based_categorization_batch(self, transactions: List[EnhancedTransaction]) -> List[EnhancedTransaction]:
        """Apply rule-based categorization to a batch"""
        for tx in transactions:
            self._apply_rule_based_categorization(tx)
        return transactions
    
    def _apply_rule_based_categorization(self, transaction: EnhancedTransaction):
        """Apply rule-based categorization to a single transaction"""
        desc_lower = transaction.description.lower()
        amount = transaction.amount
        
        # Initialize defaults
        transaction.category = 'other'
        transaction.subcategory = 'uncategorized'
        transaction.spending_type = 'regular'
        transaction.confidence = 0.6
        transaction.llm_reasoning = 'Rule-based categorization'
        
        # Income categorization (positive amounts)
        if amount > 0:
            transaction.category = 'income'
            if any(word in desc_lower for word in ['salary', 'wage', 'payroll', 'pay']):
                transaction.subcategory = 'salary'
            elif any(word in desc_lower for word in ['dividend', 'interest', 'investment']):
                transaction.subcategory = 'investment_returns'
            elif any(word in desc_lower for word in ['freelance', 'contract', 'business']):
                transaction.subcategory = 'business_income'
            else:
                transaction.subcategory = 'other_income'
            return
        
        # Expense categorization (negative amounts)
        amount_abs = abs(amount)
        
        # Determine spending behavior based on amount
        if amount_abs > 1000:
            transaction.spending_type = 'major_purchase'
        elif any(word in desc_lower for word in ['subscription', 'monthly', 'annual']):
            transaction.spending_type = 'recurring'
        
        # Category matching
        for category, keywords in self.category_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                transaction.category = category
                transaction.confidence = 0.8
                break
        
        # Enhanced subcategory detection
        self._detect_subcategory(transaction, desc_lower)
    
    def _detect_subcategory(self, transaction: EnhancedTransaction, desc_lower: str):
        """Detect subcategory based on description"""
        category = transaction.category
        
        if category == 'food_dining':
            if any(word in desc_lower for word in ['grocery', 'supermarket', 'market']):
                transaction.subcategory = 'groceries'
            elif any(word in desc_lower for word in ['restaurant', 'cafe', 'dine']):
                transaction.subcategory = 'restaurants'
            elif any(word in desc_lower for word in ['coffee', 'starbucks', 'dunkin']):
                transaction.subcategory = 'coffee_snacks'
            elif any(word in desc_lower for word in ['uber eats', 'doordash', 'delivery']):
                transaction.subcategory = 'meal_delivery'
            elif any(word in desc_lower for word in ['bar', 'liquor', 'wine', 'beer']):
                transaction.subcategory = 'alcohol'
        
        elif category == 'transportation':
            if any(word in desc_lower for word in ['gas', 'fuel', 'station']):
                transaction.subcategory = 'fuel'
            elif any(word in desc_lower for word in ['uber', 'lyft', 'taxi']):
                transaction.subcategory = 'rideshare'
            elif any(word in desc_lower for word in ['metro', 'bus', 'train', 'transit']):
                transaction.subcategory = 'public_transport'
            elif any(word in desc_lower for word in ['parking', 'garage']):
                transaction.subcategory = 'parking'
            elif any(word in desc_lower for word in ['repair', 'maintenance', 'oil change']):
                transaction.subcategory = 'vehicle_maintenance'
        
        elif category == 'fixed_expenses':
            if any(word in desc_lower for word in ['rent', 'mortgage']):
                transaction.subcategory = 'rent_mortgage'
            elif any(word in desc_lower for word in ['insurance']):
                transaction.subcategory = 'insurance'
            elif any(word in desc_lower for word in ['electric', 'gas', 'water', 'utility']):
                transaction.subcategory = 'utilities'
            elif any(word in desc_lower for word in ['phone', 'internet', 'cable']):
                transaction.subcategory = 'utilities'
            elif any(word in desc_lower for word in ['loan', 'payment', 'credit']):
                transaction.subcategory = 'loan_payments'
        
        # Add more subcategory detection as needed...
    
    def _apply_fallback_categorization(self, transaction: EnhancedTransaction):
        """Apply basic fallback categorization"""
        transaction.category = 'other'
        transaction.subcategory = 'uncategorized'
        transaction.spending_type = 'regular'
        transaction.confidence = 0.3
        transaction.llm_reasoning = 'Fallback categorization applied'
    
    def _basic_categorization(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """Most basic categorization as last resort"""
        for tx in transactions:
            if tx.amount > 0:
                tx.category = 'income'
                tx.subcategory = 'other_income'
            else:
                tx.category = 'other'
                tx.subcategory = 'uncategorized'
            
            tx.spending_type = 'regular'
            tx.confidence = 0.2
            tx.llm_reasoning = 'Basic categorization applied'
        
        analysis_result = LLMAnalysisResult(
            success=True,
            data={'method': 'basic', 'total_processed': len(transactions)},
            confidence='low',
            reasoning='Applied basic income/expense categorization'
        )
        
        return transactions, analysis_result
    
    def _calculate_confidence_distribution(self, transactions: List[EnhancedTransaction]) -> Dict:
        """Calculate confidence distribution across transactions"""
        high_confidence = sum(1 for tx in transactions if tx.confidence >= 0.8)
        medium_confidence = sum(1 for tx in transactions if 0.5 <= tx.confidence < 0.8)
        low_confidence = sum(1 for tx in transactions if tx.confidence < 0.5)
        
        total = len(transactions)
        return {
            'high': high_confidence / total if total > 0 else 0,
            'medium': medium_confidence / total if total > 0 else 0,
            'low': low_confidence / total if total > 0 else 0,
            'average_confidence': sum(tx.confidence for tx in transactions) / total if total > 0 else 0
        }
    
    def get_categorization_summary(self, transactions: List[EnhancedTransaction]) -> Dict:
        """Generate comprehensive categorization summary"""
        if not transactions:
            return {}
        
        # Category breakdown
        category_counts = {}
        subcategory_counts = {}
        spending_behavior_counts = {}
        
        total_spending_by_category = {}
        
        for tx in transactions:
            # Category counts
            category_counts[tx.category] = category_counts.get(tx.category, 0) + 1
            
            # Subcategory counts  
            subcat_key = f"{tx.category}_{tx.subcategory}"
            subcategory_counts[subcat_key] = subcategory_counts.get(subcat_key, 0) + 1
            
            # Spending behavior
            spending_behavior_counts[tx.spending_type] = spending_behavior_counts.get(tx.spending_type, 0) + 1
            
            # Spending amounts by category (only expenses)
            if tx.amount < 0:
                total_spending_by_category[tx.category] = total_spending_by_category.get(tx.category, 0) + abs(tx.amount)
        
        # Confidence statistics
        confidence_stats = self._calculate_confidence_distribution(transactions)
        
        return {
            'total_transactions': len(transactions),
            'category_breakdown': {
                'counts': category_counts,
                'spending_amounts': total_spending_by_category
            },
            'subcategory_breakdown': subcategory_counts,
            'spending_behaviors': spending_behavior_counts,
            'confidence_statistics': confidence_stats,
            'top_spending_categories': sorted(
                total_spending_by_category.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def recategorize_transaction(self, transaction: EnhancedTransaction, new_category: str, new_subcategory: str = None) -> EnhancedTransaction:
        """Manually recategorize a single transaction"""
        transaction.category = new_category
        
        if new_subcategory:
            transaction.subcategory = new_subcategory
        else:
            # Set default subcategory for the new category
            if new_category in self.category_schema:
                transaction.subcategory = self.category_schema[new_category]['subcategories'][0]
        
        transaction.confidence = 1.0  # Manual categorization is 100% confident
        transaction.llm_reasoning = 'Manually recategorized by user'
        
        return transaction


class CategorizationTrainer:
    """Train and improve categorization accuracy"""
    
    def __init__(self):
        self.training_data = []
        self.custom_rules = {}
    
    def add_training_example(self, description: str, amount: float, correct_category: str, correct_subcategory: str):
        """Add a training example for improving categorization"""
        self.training_data.append({
            'description': description.lower(),
            'amount': amount,
            'category': correct_category,
            'subcategory': correct_subcategory,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_custom_rules(self) -> Dict:
        """Generate custom categorization rules from training data"""
        rules = {}
        
        for example in self.training_data:
            desc = example['description']
            category = example['category']
            
            # Extract keywords from description
            keywords = self._extract_keywords(desc)
            
            for keyword in keywords:
                if keyword not in rules:
                    rules[keyword] = {}
                
                if category not in rules[keyword]:
                    rules[keyword][category] = 0
                
                rules[keyword][category] += 1
        
        # Keep only high-confidence rules
        filtered_rules = {}
        for keyword, categories in rules.items():
            total_occurrences = sum(categories.values())
            if total_occurrences >= 3:  # Minimum occurrences threshold
                best_category = max(categories, key=categories.get)
                confidence = categories[best_category] / total_occurrences
                
                if confidence >= 0.8:  # High confidence threshold
                    filtered_rules[keyword] = {
                        'category': best_category,
                        'confidence': confidence,
                        'occurrences': total_occurrences
                    }
        
        self.custom_rules = filtered_rules
        return filtered_rules
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract meaningful keywords from transaction description"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'until', 'without', 'within'}
        
        # Clean and split description
        words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
        keywords = [word for word in words if word not in stop_words and len(word) >= 3]
        
        return keywords[:5]  # Return top 5 keywords