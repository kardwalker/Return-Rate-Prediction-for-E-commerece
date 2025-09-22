"""
LangGraph Agent Architecture for E-commerce Return Prevention System
====================================================================

This module implements a multi-agent system using LangGraph that:
1. Predicts return probability using the trained neural network
2. Analyzes customer sentiment using Hugging Face transformers
3. Notifies customers and sellers when return risk is high
4. Provides recommendations to reduce return likelihood

Architecture Components:
- Data Collection Agent
- Return Prediction Agent  
- Sentiment Analysis Agent
- Notification Agent
- Recommendation Agent
- Workflow Orchestrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangGraph and LangChain imports
try:
    from langgraph.graph import Graph, StateGraph
    from langgraph.prebuilt import ToolExecutor
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.tools import BaseTool
    from langchain.llms.base import LLM
    print("✅ LangGraph imports successful")
except ImportError:
    print("⚠️  LangGraph not installed. Installing required packages...")
    print("Run: pip install langgraph langchain")

# Hugging Face transformers for sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    print("✅ Transformers imports successful")
except ImportError:
    print("⚠️  Transformers not installed. Installing...")
    print("Run: pip install transformers torch")

# TensorFlow for return prediction
try:
    import tensorflow as tf
    from tensorflow import keras
    print("✅ TensorFlow imports successful")
except ImportError:
    print("⚠️  TensorFlow not installed. Please install for return prediction model")

print("🤖 LangGraph Agent Architecture for Return Prevention")
print("=" * 60)

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AgentType(Enum):
    """Types of agents in the system"""
    DATA_COLLECTOR = "data_collector"
    RETURN_PREDICTOR = "return_predictor"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    NOTIFICATION_AGENT = "notification_agent"
    RECOMMENDATION_AGENT = "recommendation_agent"
    ORCHESTRATOR = "orchestrator"

@dataclass
class CustomerData:
    """Customer information structure"""
    customer_id: str
    past_return_rate: float
    purchase_history: List[Dict]
    demographics: Dict[str, Any]
    preferences: Dict[str, Any]
    feedback_history: List[str]

@dataclass
class ProductData:
    """Product information structure"""
    product_id: str
    category: str
    brand: str
    price: float
    seller_rating: float
    material: str
    size_info: Dict[str, Any]
    image_quality_score: float
    stock_type: str

@dataclass
class SellerData:
    """Seller information structure"""
    seller_id: str
    rating: float
    return_rate: float
    response_time: float
    quality_score: float
    contact_info: Dict[str, str]

@dataclass
class TransactionContext:
    """Current transaction context"""
    customer: CustomerData
    product: ProductData
    seller: SellerData
    order_details: Dict[str, Any]
    payment_method: str
    timestamp: datetime

@dataclass
class PredictionResult:
    """Return prediction results"""
    return_probability: float
    confidence_score: float
    risk_factors: List[str]
    alert_level: AlertLevel
    recommendations: List[str]

@dataclass
class SentimentResult:
    """Sentiment analysis results"""
    sentiment_score: float  # -1 (negative) to 1 (positive)
    sentiment_label: str   # positive, negative, neutral
    confidence: float
    key_emotions: List[str]
    concern_areas: List[str]

class AgentState:
    """Shared state between agents"""
    def __init__(self):
        self.transaction_context: Optional[TransactionContext] = None
        self.prediction_result: Optional[PredictionResult] = None
        self.sentiment_result: Optional[SentimentResult] = None
        self.notifications_sent: List[Dict] = []
        self.recommendations: List[str] = []
        self.agent_messages: List[BaseMessage] = []
        self.workflow_status: str = "initialized"

class DataCollectorAgent:
    """Agent responsible for collecting and preprocessing data"""
    
    def __init__(self):
        self.name = "DataCollector"
        
    def collect_customer_data(self, customer_id: str) -> CustomerData:
        """Collect customer data from various sources"""
        # In real implementation, this would query databases
        # For demo, we'll simulate realistic data
        
        return CustomerData(
            customer_id=customer_id,
            past_return_rate=np.random.uniform(0.1, 0.4),
            purchase_history=[
                {"product_id": f"prod_{i}", "returned": np.random.choice([0, 1], p=[0.7, 0.3])}
                for i in range(10)
            ],
            demographics={
                "age_group": np.random.choice(["18-25", "26-35", "36-45", "46+"]),
                "city_tier": np.random.choice(["metro", "tier-1", "tier-2", "tier-3"]),
                "gender": np.random.choice(["male", "female", "other"])
            },
            preferences={
                "categories": ["fashion", "electronics"],
                "brands": ["premium", "mid-tier"],
                "price_sensitivity": np.random.uniform(0.2, 0.8)
            },
            feedback_history=[
                "Good quality product, fast delivery",
                "Size was smaller than expected",
                "Material quality could be better"
            ]
        )
    
    def collect_product_data(self, product_id: str) -> ProductData:
        """Collect product information"""
        
        categories = ["Jeans", "T-shirt", "Dress", "Shoes", "Accessories"]
        brands = ["Premium Brand", "Mid-tier Brand", "Budget Brand"]
        materials = ["Cotton", "Polyester", "Silk", "Denim", "Leather"]
        
        return ProductData(
            product_id=product_id,
            category=np.random.choice(categories),
            brand=np.random.choice(brands),
            price=np.random.uniform(500, 3000),
            seller_rating=np.random.uniform(3.0, 5.0),
            material=np.random.choice(materials),
            size_info={"available_sizes": ["S", "M", "L", "XL"], "size_chart_available": True},
            image_quality_score=np.random.uniform(6, 10),
            stock_type=np.random.choice(["fast-fashion", "evergreen"])
        )
    
    def collect_seller_data(self, seller_id: str) -> SellerData:
        """Collect seller information"""
        
        return SellerData(
            seller_id=seller_id,
            rating=np.random.uniform(3.5, 5.0),
            return_rate=np.random.uniform(0.15, 0.35),
            response_time=np.random.uniform(1, 24),  # hours
            quality_score=np.random.uniform(0.7, 1.0),
            contact_info={
                "email": f"{seller_id}@seller.com",
                "phone": "+91-XXXXXXXXXX"
            }
        )
    
    def execute(self, state: AgentState, transaction_data: Dict) -> AgentState:
        """Execute data collection"""
        
        print(f"🔍 {self.name}: Collecting transaction data...")
        
        # Collect all necessary data
        customer = self.collect_customer_data(transaction_data["customer_id"])
        product = self.collect_product_data(transaction_data["product_id"])
        seller = self.collect_seller_data(transaction_data["seller_id"])
        
        # Create transaction context
        state.transaction_context = TransactionContext(
            customer=customer,
            product=product,
            seller=seller,
            order_details=transaction_data.get("order_details", {}),
            payment_method=transaction_data.get("payment_method", "COD"),
            timestamp=datetime.now()
        )
        
        state.agent_messages.append(
            AIMessage(content=f"Data collection completed for transaction {transaction_data['product_id']}")
        )
        
        print(f"✅ Data collection completed")
        return state

class ReturnPredictorAgent:
    """Agent for predicting return probability using neural network"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.name = "ReturnPredictor"
        self.model = None
        self.feature_scaler = None
        self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str]):
        """Load trained return prediction model"""
        try:
            if model_path:
                self.model = keras.models.load_model(model_path)
                print(f"✅ Loaded return prediction model from {model_path}")
            else:
                print(f"⚠️  No model path provided. Using mock predictions.")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}. Using mock predictions.")
    
    def extract_features(self, context: TransactionContext) -> np.ndarray:
        """Extract features from transaction context for prediction"""
        
        # This would normally use the same feature extraction as training
        # For demo, we'll create a feature vector
        
        features = [
            # Customer features
            context.customer.past_return_rate,
            len(context.customer.purchase_history),
            context.customer.preferences.get("price_sensitivity", 0.5),
            
            # Product features  
            context.product.price,
            context.product.seller_rating,
            context.product.image_quality_score,
            1 if context.product.stock_type == "fast-fashion" else 0,
            
            # Seller features
            context.seller.rating,
            context.seller.return_rate,
            context.seller.response_time,
            
            # Order features
            1 if context.payment_method == "COD" else 0,
            context.timestamp.hour,  # Order hour
            
            # Additional mock features to match expected dimension
        ] + [np.random.normal(0, 1) for _ in range(23)]  # Pad to 35 features
        
        return np.array(features).reshape(1, -1)
    
    def assess_risk_factors(self, context: TransactionContext, probability: float) -> List[str]:
        """Identify key risk factors contributing to high return probability"""
        
        risk_factors = []
        
        if context.customer.past_return_rate > 0.3:
            risk_factors.append("High customer return history")
        
        if context.product.seller_rating < 4.0:
            risk_factors.append("Low seller rating")
        
        if context.payment_method == "COD":
            risk_factors.append("Cash on Delivery payment")
        
        if context.product.stock_type == "fast-fashion":
            risk_factors.append("Fast fashion item")
        
        if context.product.image_quality_score < 7:
            risk_factors.append("Poor product images")
        
        if context.seller.return_rate > 0.25:
            risk_factors.append("Seller has high return rate")
        
        return risk_factors
    
    def determine_alert_level(self, probability: float) -> AlertLevel:
        """Determine alert level based on return probability"""
        
        if probability >= 0.7:
            return AlertLevel.CRITICAL
        elif probability >= 0.5:
            return AlertLevel.HIGH
        elif probability >= 0.3:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute return prediction"""
        
        print(f"🧠 {self.name}: Predicting return probability...")
        
        if not state.transaction_context:
            raise ValueError("Transaction context not available")
        
        # Extract features
        features = self.extract_features(state.transaction_context)
        
        # Make prediction
        if self.model:
            probability = float(self.model.predict(features, verbose=0)[0][0])
            confidence = 0.85  # Could be computed from model uncertainty
        else:
            # Mock prediction for demo
            probability = np.random.uniform(0.1, 0.8)
            confidence = 0.75
        
        # Assess risk factors
        risk_factors = self.assess_risk_factors(state.transaction_context, probability)
        
        # Determine alert level
        alert_level = self.determine_alert_level(probability)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(state.transaction_context, risk_factors)
        
        # Store prediction result
        state.prediction_result = PredictionResult(
            return_probability=probability,
            confidence_score=confidence,
            risk_factors=risk_factors,
            alert_level=alert_level,
            recommendations=recommendations
        )
        
        state.agent_messages.append(
            AIMessage(content=f"Return probability predicted: {probability:.1%} (Alert: {alert_level.value})")
        )
        
        print(f"✅ Prediction: {probability:.1%} probability, {alert_level.value} alert")
        return state
    
    def generate_recommendations(self, context: TransactionContext, risk_factors: List[str]) -> List[str]:
        """Generate recommendations to reduce return risk"""
        
        recommendations = []
        
        if "Poor product images" in risk_factors:
            recommendations.append("Improve product images and add multiple angles")
        
        if "Cash on Delivery payment" in risk_factors:
            recommendations.append("Encourage prepaid payment with discount")
        
        if "Low seller rating" in risk_factors:
            recommendations.append("Provide additional quality assurance")
        
        if "High customer return history" in risk_factors:
            recommendations.append("Offer virtual try-on or size consultation")
        
        return recommendations

class SentimentAnalyzerAgent:
    """Agent for analyzing customer sentiment using Hugging Face transformers"""
    
    def __init__(self):
        self.name = "SentimentAnalyzer"
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.load_models()
    
    def load_models(self):
        """Load sentiment analysis models"""
        try:
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load emotion analysis pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            print(f"✅ Sentiment analysis models loaded")
            
        except Exception as e:
            print(f"⚠️  Could not load sentiment models: {e}")
            print("Using mock sentiment analysis")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of customer feedback"""
        
        if self.sentiment_pipeline:
            # Real sentiment analysis
            sentiment_results = self.sentiment_pipeline(text)
            
            # Convert to unified format
            sentiment_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
            best_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
            
            sentiment_label = sentiment_map.get(best_sentiment['label'], best_sentiment['label'].lower())
            sentiment_score = best_sentiment['score'] if sentiment_label == "positive" else -best_sentiment['score']
            
        else:
            # Mock sentiment analysis
            sentiment_score = np.random.uniform(-1, 1)
            sentiment_label = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence": abs(sentiment_score)
        }
    
    def analyze_emotions(self, text: str) -> List[str]:
        """Analyze emotions in customer feedback"""
        
        if self.emotion_pipeline:
            emotion_results = self.emotion_pipeline(text)
            # Get top emotions
            top_emotions = sorted(emotion_results[0], key=lambda x: x['score'], reverse=True)[:3]
            return [emotion['label'] for emotion in top_emotions]
        else:
            # Mock emotions
            emotions = ["joy", "anger", "sadness", "fear", "surprise", "disgust"]
            return np.random.choice(emotions, 2, replace=False).tolist()
    
    def identify_concerns(self, feedback_history: List[str]) -> List[str]:
        """Identify concern areas from feedback history"""
        
        concerns = []
        
        # Simple keyword-based concern detection (could be enhanced with NLP)
        concern_keywords = {
            "size": ["size", "fit", "small", "large", "tight", "loose"],
            "quality": ["quality", "cheap", "flimsy", "durable", "material"],
            "delivery": ["delivery", "shipping", "late", "damaged", "packaging"],
            "service": ["service", "support", "response", "rude", "helpful"],
            "description": ["description", "different", "expected", "misleading"]
        }
        
        feedback_text = " ".join(feedback_history).lower()
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in feedback_text for keyword in keywords):
                concerns.append(concern)
        
        return concerns
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute sentiment analysis"""
        
        print(f"💭 {self.name}: Analyzing customer sentiment...")
        
        if not state.transaction_context:
            raise ValueError("Transaction context not available")
        
        customer = state.transaction_context.customer
        
        # Analyze sentiment from feedback history
        combined_feedback = " ".join(customer.feedback_history)
        
        if combined_feedback.strip():
            sentiment_data = self.analyze_sentiment(combined_feedback)
            emotions = self.analyze_emotions(combined_feedback)
            concerns = self.identify_concerns(customer.feedback_history)
        else:
            # No feedback available
            sentiment_data = {"sentiment_score": 0, "sentiment_label": "neutral", "confidence": 0.5}
            emotions = []
            concerns = []
        
        # Store sentiment result
        state.sentiment_result = SentimentResult(
            sentiment_score=sentiment_data["sentiment_score"],
            sentiment_label=sentiment_data["sentiment_label"],
            confidence=sentiment_data["confidence"],
            key_emotions=emotions,
            concern_areas=concerns
        )
        
        state.agent_messages.append(
            AIMessage(content=f"Sentiment analysis: {sentiment_data['sentiment_label']} ({sentiment_data['sentiment_score']:.2f})")
        )
        
        print(f"✅ Sentiment: {sentiment_data['sentiment_label']} (score: {sentiment_data['sentiment_score']:.2f})")
        return state

class NotificationAgent:
    """Agent for sending notifications to customers and sellers"""
    
    def __init__(self):
        self.name = "NotificationAgent"
    
    def create_customer_notification(self, context: TransactionContext, 
                                   prediction: PredictionResult, 
                                   sentiment: SentimentResult) -> Dict[str, Any]:
        """Create customer notification message"""
        
        if prediction.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            # High risk - warn customer
            message = {
                "type": "risk_warning",
                "title": "⚠️ Important Product Information",
                "content": f"""
Hi! We want to ensure you're completely satisfied with your purchase.

Based on your preferences and this product's characteristics, we've identified some areas that might affect your satisfaction:

Risk Factors:
{chr(10).join(f"• {factor}" for factor in prediction.risk_factors)}

Recommendations:
{chr(10).join(f"• {rec}" for rec in prediction.recommendations)}

Would you like to:
- Review size guide and product details
- Chat with our style advisor
- Consider similar products with better fit probability
- Proceed with additional quality assurance

Your satisfaction is our priority! 😊
                """,
                "urgency": prediction.alert_level.value,
                "actions": ["review_details", "chat_advisor", "see_alternatives", "proceed_assured"]
            }
        else:
            # Low risk - positive reinforcement
            message = {
                "type": "positive_reinforcement", 
                "title": "Great Choice! 🎉",
                "content": f"""
This product looks like a perfect match for you based on your preferences and purchase history!

✅ High satisfaction probability
✅ Matches your style preferences  
✅ Good seller ratings
✅ Quality assurance

Enjoy your purchase!
                """,
                "urgency": "low",
                "actions": ["proceed_checkout"]
            }
        
        return message
    
    def create_seller_notification(self, context: TransactionContext, 
                                 prediction: PredictionResult) -> Dict[str, Any]:
        """Create seller notification message"""
        
        if prediction.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            message = {
                "type": "high_risk_alert",
                "title": f"🚨 High Return Risk Alert - Order #{context.product.product_id}",
                "content": f"""
High return probability detected: {prediction.return_probability:.1%}

Risk Factors:
{chr(10).join(f"• {factor}" for factor in prediction.risk_factors)}

Recommended Actions:
{chr(10).join(f"• {rec}" for rec in prediction.recommendations)}

Customer Profile:
- Past return rate: {context.customer.past_return_rate:.1%}
- Sentiment: {context.customer.demographics.get('city_tier', 'Unknown')}

Please take preventive measures to ensure customer satisfaction.
                """,
                "urgency": prediction.alert_level.value,
                "actions": ["contact_customer", "improve_listing", "quality_check"]
            }
        else:
            message = {
                "type": "standard_order",
                "title": f"📦 New Order - {context.product.product_id}",
                "content": f"Low return risk order. Standard processing recommended.",
                "urgency": "low",
                "actions": ["process_order"]
            }
        
        return message
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute notification sending"""
        
        print(f"📨 {self.name}: Sending notifications...")
        
        if not all([state.transaction_context, state.prediction_result]):
            raise ValueError("Required data not available for notifications")
        
        # Create notifications
        customer_notification = self.create_customer_notification(
            state.transaction_context, 
            state.prediction_result, 
            state.sentiment_result
        )
        
        seller_notification = self.create_seller_notification(
            state.transaction_context,
            state.prediction_result
        )
        
        # Store notifications (in real system, would send via email/SMS/push)
        state.notifications_sent = [
            {
                "recipient_type": "customer",
                "recipient_id": state.transaction_context.customer.customer_id,
                "notification": customer_notification,
                "timestamp": datetime.now()
            },
            {
                "recipient_type": "seller", 
                "recipient_id": state.transaction_context.seller.seller_id,
                "notification": seller_notification,
                "timestamp": datetime.now()
            }
        ]
        
        state.agent_messages.append(
            AIMessage(content=f"Notifications sent to customer and seller ({state.prediction_result.alert_level.value} priority)")
        )
        
        print(f"✅ Notifications sent - Alert Level: {state.prediction_result.alert_level.value}")
        return state

class RecommendationAgent:
    """Agent for generating personalized recommendations"""
    
    def __init__(self):
        self.name = "RecommendationAgent"
    
    def generate_alternative_products(self, context: TransactionContext, 
                                    risk_factors: List[str]) -> List[Dict[str, Any]]:
        """Generate alternative product recommendations"""
        
        alternatives = []
        
        # Based on risk factors, suggest better alternatives
        if "Low seller rating" in risk_factors:
            alternatives.append({
                "type": "better_seller",
                "description": "Same product from higher-rated seller",
                "benefit": "Higher seller rating (4.8/5)",
                "product_id": f"alt_seller_{context.product.product_id}"
            })
        
        if "Poor product images" in risk_factors:
            alternatives.append({
                "type": "better_images",
                "description": "Similar product with detailed images",
                "benefit": "360° view and size guide available",
                "product_id": f"alt_images_{context.product.product_id}"
            })
        
        if "Fast fashion item" in risk_factors:
            alternatives.append({
                "type": "better_quality",
                "description": "Premium version of similar product",
                "benefit": "Better material and durability",
                "product_id": f"alt_premium_{context.product.product_id}"
            })
        
        return alternatives
    
    def generate_size_recommendations(self, context: TransactionContext) -> List[str]:
        """Generate size-related recommendations"""
        
        recommendations = []
        
        if context.customer.past_return_rate > 0.25:
            recommendations.extend([
                "Use our AI size predictor tool",
                "Check detailed size chart and measurements",
                "Read customer reviews for sizing feedback",
                "Consider ordering multiple sizes (free returns)"
            ])
        
        return recommendations
    
    def generate_experience_improvements(self, sentiment: SentimentResult) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        
        improvements = []
        
        if "size" in sentiment.concern_areas:
            improvements.append("Provide virtual fitting room experience")
        
        if "quality" in sentiment.concern_areas:
            improvements.append("Show detailed material and quality certificates")
        
        if "delivery" in sentiment.concern_areas:
            improvements.append("Offer premium packaging and faster delivery")
        
        if "service" in sentiment.concern_areas:
            improvements.append("Assign dedicated customer success manager")
        
        return improvements
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute recommendation generation"""
        
        print(f"💡 {self.name}: Generating recommendations...")
        
        if not all([state.transaction_context, state.prediction_result]):
            raise ValueError("Required data not available for recommendations")
        
        # Generate different types of recommendations
        alternative_products = self.generate_alternative_products(
            state.transaction_context, 
            state.prediction_result.risk_factors
        )
        
        size_recommendations = self.generate_size_recommendations(state.transaction_context)
        
        experience_improvements = []
        if state.sentiment_result:
            experience_improvements = self.generate_experience_improvements(state.sentiment_result)
        
        # Combine all recommendations
        all_recommendations = {
            "alternative_products": alternative_products,
            "size_guidance": size_recommendations,
            "experience_improvements": experience_improvements,
            "risk_mitigation": state.prediction_result.recommendations
        }
        
        state.recommendations = all_recommendations
        
        state.agent_messages.append(
            AIMessage(content=f"Generated {len(alternative_products)} alternatives and {len(size_recommendations)} size recommendations")
        )
        
        print(f"✅ Recommendations generated")
        return state

def create_workflow_diagram():
    """Create and display the LangGraph workflow diagram"""
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (agents)
    agents = [
        ("Start", {"type": "start", "color": "lightgreen"}),
        ("Data Collector", {"type": "agent", "color": "lightblue"}),
        ("Return Predictor", {"type": "agent", "color": "orange"}),
        ("Sentiment Analyzer", {"type": "agent", "color": "pink"}),
        ("Notification Agent", {"type": "agent", "color": "yellow"}),
        ("Recommendation Agent", {"type": "agent", "color": "lightcoral"}),
        ("Decision Point", {"type": "decision", "color": "red"}),
        ("End", {"type": "end", "color": "lightgray"})
    ]
    
    for agent, attrs in agents:
        G.add_node(agent, **attrs)
    
    # Add edges (workflow flow)
    edges = [
        ("Start", "Data Collector"),
        ("Data Collector", "Return Predictor"),
        ("Data Collector", "Sentiment Analyzer"), 
        ("Return Predictor", "Decision Point"),
        ("Sentiment Analyzer", "Decision Point"),
        ("Decision Point", "Notification Agent"),
        ("Decision Point", "Recommendation Agent"),
        ("Notification Agent", "End"),
        ("Recommendation Agent", "End")
    ]
    
    G.add_edges_from(edges)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Layout
    pos = {
        "Start": (0, 0),
        "Data Collector": (1, 0),
        "Return Predictor": (2, 1),
        "Sentiment Analyzer": (2, -1),
        "Decision Point": (3, 0),
        "Notification Agent": (4, 1),
        "Recommendation Agent": (4, -1),
        "End": (5, 0)
    }
    
    # Draw nodes
    for node_type in ["start", "agent", "decision", "end"]:
        nodes = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
        colors = [G.nodes[n]["color"] for n in nodes]
        
        if node_type == "decision":
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                 node_shape="D", node_size=2000)
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                 node_size=3000)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, 
                          arrowsize=20, arrowstyle="-|>")
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add detailed annotations
    plt.text(0, -0.5, "Transaction\nInitiated", ha="center", va="center", 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.text(1, -0.5, "• Customer Data\n• Product Data\n• Seller Data", 
             ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.text(2, 1.5, "Neural Network\nReturn Prediction", ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    plt.text(2, -1.5, "HuggingFace\nSentiment Analysis", ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="pink", alpha=0.7))
    
    plt.text(3, -0.5, "Risk Level\nAssessment", ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    plt.text(4, 1.5, "• Customer Alert\n• Seller Alert", ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.text(4, -1.5, "• Alternatives\n• Size Guide\n• Improvements", ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.text(5, -0.5, "Process\nComplete", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.title("LangGraph Agent Architecture for E-commerce Return Prevention\n" +
              "🤖 Multi-Agent System with Neural Network & NLP Integration", 
              fontsize=14, fontweight="bold", pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=15, label='Data Processing Agent'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=15, label='ML Prediction Agent'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', 
                   markersize=15, label='NLP Analysis Agent'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
                   markersize=12, label='Decision Point'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markersize=15, label='Notification Agent'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=15, label='Recommendation Agent')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('langgraph_workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Workflow diagram saved as 'langgraph_workflow_diagram.png'")

class WorkflowOrchestrator:
    """Main orchestrator for the LangGraph workflow"""
    
    def __init__(self):
        self.agents = {
            AgentType.DATA_COLLECTOR: DataCollectorAgent(),
            AgentType.RETURN_PREDICTOR: ReturnPredictorAgent(),
            AgentType.SENTIMENT_ANALYZER: SentimentAnalyzerAgent(), 
            AgentType.NOTIFICATION_AGENT: NotificationAgent(),
            AgentType.RECOMMENDATION_AGENT: RecommendationAgent()
        }
        
    def create_workflow_graph(self):
        """Create LangGraph workflow"""
        
        # This would use actual LangGraph syntax in real implementation
        # For demo, we'll simulate the workflow
        
        workflow_steps = [
            AgentType.DATA_COLLECTOR,
            AgentType.RETURN_PREDICTOR,
            AgentType.SENTIMENT_ANALYZER,
            AgentType.NOTIFICATION_AGENT,
            AgentType.RECOMMENDATION_AGENT
        ]
        
        return workflow_steps
    
    def execute_workflow(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete workflow"""
        
        print(f"🚀 Starting LangGraph Workflow for Transaction: {transaction_data.get('product_id', 'Unknown')}")
        print("=" * 70)
        
        # Initialize state
        state = AgentState()
        
        # Execute workflow steps
        workflow_steps = self.create_workflow_graph()
        
        try:
            # Step 1: Data Collection
            state = self.agents[AgentType.DATA_COLLECTOR].execute(state, transaction_data)
            
            # Step 2 & 3: Parallel execution of prediction and sentiment analysis
            state = self.agents[AgentType.RETURN_PREDICTOR].execute(state)
            state = self.agents[AgentType.SENTIMENT_ANALYZER].execute(state)
            
            # Decision point: Check if high risk
            if state.prediction_result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
                print(f"⚠️  HIGH RISK DETECTED - Triggering alerts and recommendations")
                
                # Step 4 & 5: Send notifications and generate recommendations
                state = self.agents[AgentType.NOTIFICATION_AGENT].execute(state)
                state = self.agents[AgentType.RECOMMENDATION_AGENT].execute(state)
            else:
                print(f"✅ LOW RISK - Standard processing")
                # Still send standard notifications
                state = self.agents[AgentType.NOTIFICATION_AGENT].execute(state)
            
            state.workflow_status = "completed"
            
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            state.workflow_status = "failed"
        
        # Compile results
        results = {
            "transaction_id": transaction_data.get("product_id"),
            "workflow_status": state.workflow_status,
            "return_probability": state.prediction_result.return_probability if state.prediction_result else None,
            "alert_level": state.prediction_result.alert_level.value if state.prediction_result else None,
            "sentiment": state.sentiment_result.sentiment_label if state.sentiment_result else None,
            "notifications_sent": len(state.notifications_sent),
            "recommendations": state.recommendations,
            "execution_summary": {
                "risk_factors": state.prediction_result.risk_factors if state.prediction_result else [],
                "concern_areas": state.sentiment_result.concern_areas if state.sentiment_result else [],
                "agent_messages": [msg.content for msg in state.agent_messages]
            }
        }
        
        print(f"\n🎯 Workflow Summary:")
        print(f"  Return Probability: {results['return_probability']:.1%}" if results['return_probability'] else "  Return Probability: N/A")
        print(f"  Alert Level: {results['alert_level']}")
        print(f"  Sentiment: {results['sentiment']}")
        print(f"  Notifications: {results['notifications_sent']}")
        print(f"  Status: {results['workflow_status']}")
        
        return results

def demo_workflow():
    """Demonstrate the complete workflow with sample data"""
    
    print("🎭 LangGraph Agent Architecture Demo")
    print("=" * 60)
    
    # Sample transaction data
    sample_transactions = [
        {
            "customer_id": "CUST_001",
            "product_id": "PROD_HIGH_RISK_001", 
            "seller_id": "SELLER_001",
            "payment_method": "COD",
            "order_details": {"quantity": 1, "size": "L", "color": "Blue"}
        },
        {
            "customer_id": "CUST_002",
            "product_id": "PROD_LOW_RISK_002",
            "seller_id": "SELLER_002", 
            "payment_method": "UPI",
            "order_details": {"quantity": 1, "size": "M", "color": "Red"}
        }
    ]
    
    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator()
    
    # Execute workflow for each transaction
    results = []
    for i, transaction in enumerate(sample_transactions, 1):
        print(f"\n{'='*20} Transaction {i} {'='*20}")
        result = orchestrator.execute_workflow(transaction)
        results.append(result)
    
    # Display summary
    print(f"\n📊 FINAL SUMMARY")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\nTransaction {i}:")
        print(f"  🎯 Return Risk: {result['alert_level']}")
        print(f"  😊 Sentiment: {result['sentiment']}")
        print(f"  📨 Notifications: {result['notifications_sent']}")
        print(f"  ⚡ Status: {result['workflow_status']}")
    
    return results

if __name__ == "__main__":
    # Create workflow diagram
    create_workflow_diagram()
    
    # Run demo
    demo_results = demo_workflow()
    
    print(f"\n🎉 LangGraph Agent Architecture Demo Complete!")
    print(f"📊 Workflow diagram saved as 'langgraph_workflow_diagram.png'")
