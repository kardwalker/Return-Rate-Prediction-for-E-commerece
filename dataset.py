print("""This script creates a comprehensive dataset for fashion e-commerce features based on the analysis of meesho.
The dataset includes product-level, basket-level, and customer-level features.
This dataset is synthetic and for demonstration purposes only."
      """)
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
from typing import Dict, List, Any

# Define the feature categories and their descriptions
feature_categories = {
    "product_level": {
        "description": "Describe each individual fashion item (order line-item)",
        "features": {
            "brand_label": "local boutique seller, unbranded, premium brand",
            "category": "saree, kurta, lehenga, western dress, t-shirt, jeans, footwear",
            "subcategory": "cotton saree vs silk saree, sneakers vs sandals",
            "size": "S, M, L, XL, free size",
            "color": "red, black, etc. — mapped to ~12 base colors",
            "price_mrp": "MRP price",
            "price_discounted": "discounted price",
            "discount_percent": "steep discount can drive impulsive orders → more returns",
            "material_fabric": "cotton, silk, polyester — mismatch with expectation may drive returns",
            "fit_type": "slim, regular, free-size",
            "seller_rating": "high-rated vs new/low-rated sellers",
            "stock_type": "fast-fashion vs evergreen SKU",
            "images_count_quality": "better images → fewer returns"
        }
    },
    "basket_level": {
        "description": "Describe the shopping session (per order/cart)",
        "features": {
            "basket_size": "number of items in cart",
            "basket_diversity": "same subcategory multiple times, e.g., ordering 4 kurtas in different colors",
            "basket_value": "total ₹ value of the order",
            "channel": "Meesho app vs web",
            "device_type": "Android low-end vs iOS vs desktop",
            "operating_system_version": "proxy for customer affluence",
            "payment_method": "COD vs prepaid vs UPI — COD tends to have higher return",
            "order_time": "late-night impulse orders vs daytime orders",
            "delivery_speed_option": "standard vs express"
        }
    },
    "customer_level": {
        "description": "Capture history & behavior of the customer (per user)",
        "features": {
            "past_return_rate": "fraction of past orders returned",
            "past_purchase_volume": "number of orders, frequency",
            "category_affinity": "mostly kurtas vs mostly western wear",
            "return_reasons_distribution": "wrong size, color mismatch, quality issue",
            "city_tier": "metro vs tier-2 vs tier-3 — COD usage higher in smaller towns",
            "average_basket_value": "low-ticket vs high-ticket shopper",
            "payment_preference_history": "COD-heavy vs prepaid-heavy",
            "fraud_indicators": "patterns like ordering expensive clothes → return after use",
            "tenure_with_platform": "new vs long-term customer"
        }
    }
}

# Create sample data for demonstration
def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate sample fashion e-commerce data"""
    np.random.seed(42)
    
    # Meesho category structure from the images
    
    # Women Ethnic categories
    women_ethnic_categories = {
        'Sarees': ['All Sarees', 'Silk Sarees', 'Banarasi Silk Sarees', 'Cotton Sarees', 
                   'Georgette Sarees', 'Chiffon Sarees', 'Heavy Work Sarees', 'Net Sarees'],
        'Kurtis': ['All Kurtis', 'Anarkali Kurtis', 'Rayon Kurtis', 'Cotton Kurtis', 'Chikankari Kurtis'],
        'Kurta Sets': ['All Kurta Sets', 'Kurta Palazzo Sets', 'Rayon Kurta Sets', 'Kurta Pant Sets', 
                       'Cotton Kurta Sets', 'Sharara Sets'],
        'Dupatta Sets': ['Cotton Sets', 'Rayon Sets', 'Printed Sets'],
        'Suits & Dress Material': ['All Suits & Dress Material', 'Cotton Suits', 'Embroidered Suits', 
                                   'Crepe Suits', 'Silk Suits', 'Patiala Suits'],
        'Lehengas': ['Lehenga Cholis', 'Net Lehenga', 'Bridal Lehenga'],
        'Other Ethnic': ['Blouses', 'Dupattas', 'Lehanga', 'Gown', 'Skirts & Bottomwear', 
                         'Islamic Fashion', 'Petticoats']
    }
    
    # Women Western categories
    women_western_categories = {
        'Tops': ['Casual tops', 'formal blouses', 'shirts'],
        'Dresses': ['Party dresses', 'casual dresses', 'formal wear'],
        'Jeans': ['Skinny jeans', 'straight fit', 'bootcut'],
        'T-shirts': ['Basic tees', 'graphic tees', 'polo shirts'],
        'Shirts': ['Casual shirts', 'formal shirts'],
        'Trousers': ['Formal pants', 'casual trousers'],
        'Shorts': ['Denim shorts', 'casual shorts'],
        'Skirts': ['Mini skirts', 'maxi skirts', 'A-line'],
        'Jumpsuits': ['Casual jumpsuits', 'formal jumpsuits'],
        'Co-ord Sets': ['Matching top and bottom sets']
    }
    
    # Men categories
    men_categories = {
        'T-shirts': ['Basic tees', 'polo shirts', 'graphic tees'],
        'Shirts': ['Casual shirts', 'formal shirts'],
        'Jeans': ['Skinny fit', 'regular fit', 'slim fit'],
        'Trousers': ['Formal pants', 'casual trousers'],
        'Shorts': ['Casual shorts', 'sports shorts'],
        'Ethnic Wear': ['Kurtas', 'sherwanis', 'ethnic jackets'],
        'Jackets': ['Casual jackets', 'formal coats'],
        'Sweatshirts': ['Hoodies', 'pullover sweatshirts'],
        'Track Suits': ['Sports wear', 'loungewear'],
        'Innerwear': ['Vests', 'briefs', 'boxers']
    }
    
    # Kids categories with age groups
    kids_categories = {
        'Boys Clothing': {
            '0-2 years': ['Rompers', 'onesies', 'sets'],
            '2-8 years': ['T-shirts', 'shorts', 'jeans'],
            '8-16 years': ['Shirts', 'trousers', 'ethnic wear']
        },
        'Girls Clothing': {
            '0-2 years': ['Frocks', 'rompers', 'sets'],
            '2-8 years': ['Dresses', 'tops', 'leggings'],
            '8-16 years': ['Kurtas', 'western wear']
        },
        'Ethnic Wear': ['Kurtas', 'lehengas', 'dhoti sets'],
        'Footwear': ['Shoes', 'sandals', 'sneakers'],
        'Accessories': ['Bags', 'hair accessories'],
        'Baby Care': ['Diapers', 'feeding accessories']
    }
    
    # Sample data generation
    brands = ['Local Boutique', 'Unbranded', 'Premium Brand', 'Mid-tier Brand']
    sizes = ['S', 'M', 'L', 'XL', 'XXL', 'Free Size', '0-3M', '3-6M', '6-12M', '1-2Y', '2-3Y', '3-4Y', '4-5Y', '6-7Y', '8-9Y', '10-11Y', '12-13Y', '14-15Y']
    colors = ['red', 'black', 'blue', 'white', 'green', 'yellow', 'pink', 'brown', 'grey', 'navy', 'maroon', 'orange']
    materials = ['cotton', 'silk', 'polyester', 'blend', 'denim', 'leather', 'rayon', 'georgette', 'chiffon', 'net']
    fit_types = ['slim', 'regular', 'loose', 'free-size', 'skinny', 'straight', 'bootcut']
    channels = ['app', 'web']
    device_types = ['Android_low', 'Android_high', 'iOS', 'desktop']
    payment_methods = ['COD', 'prepaid', 'UPI']
    city_tiers = ['metro', 'tier-2', 'tier-3']
    past_return_reasons = ['wrong_size', 'color_mismatch', 'quality_issue', 'damaged', 'not_as_described']
    
    data = []
    
    def select_category_and_pricing():
        """Select category and determine pricing based on target demographic"""
        # Randomly select target demographic
        demographics = ['women_ethnic', 'women_western', 'men', 'kids']
        target_demo = np.random.choice(demographics)
        
        if target_demo == 'women_ethnic':
            main_category = np.random.choice(list(women_ethnic_categories.keys()))
            subcategory = np.random.choice(women_ethnic_categories[main_category])
            
            # Price ranges for women ethnic wear
            if main_category in ['Sarees', 'Lehengas']:
                base_price = np.random.uniform(899, 8999)  # Higher for sarees/lehengas
            elif main_category in ['Suits & Dress Material']:
                base_price = np.random.uniform(699, 4999)
            else:
                base_price = np.random.uniform(399, 2999)
                
        elif target_demo == 'women_western':
            main_category = np.random.choice(list(women_western_categories.keys()))
            subcategory = np.random.choice(women_western_categories[main_category])
            
            # Price ranges for women western wear
            if main_category in ['Dresses', 'Jumpsuits']:
                base_price = np.random.uniform(599, 3999)
            elif main_category in ['Jeans', 'Trousers']:
                base_price = np.random.uniform(499, 2499)
            else:
                base_price = np.random.uniform(299, 1999)
                
        elif target_demo == 'men':
            main_category = np.random.choice(list(men_categories.keys()))
            subcategory = np.random.choice(men_categories[main_category])
            
            # Price ranges for men's wear
            if main_category in ['Ethnic Wear', 'Jackets']:
                base_price = np.random.uniform(699, 4999)
            elif main_category in ['Shirts', 'Jeans', 'Trousers']:
                base_price = np.random.uniform(499, 2999)
            else:
                base_price = np.random.uniform(199, 1499)
                
        else:  # kids
            main_category = np.random.choice(list(kids_categories.keys()))
            if main_category in ['Boys Clothing', 'Girls Clothing']:
                age_group = np.random.choice(list(kids_categories[main_category].keys()))
                subcategory = np.random.choice(kids_categories[main_category][age_group])
                
                # Kids pricing - significantly lower than adult clothing
                if age_group == '0-2 years':
                    base_price = np.random.uniform(149, 699)  # Baby clothes
                elif age_group == '2-8 years':
                    base_price = np.random.uniform(199, 999)  # Toddler/young kids
                else:  # 8-16 years
                    base_price = np.random.uniform(299, 1499)  # Older kids
            else:
                subcategory = np.random.choice(kids_categories[main_category]) if isinstance(kids_categories[main_category], list) else main_category
                age_group = f"{np.random.choice(['0-2', '2-8', '8-16'])} years"
                base_price = np.random.uniform(199, 999)  # General kids items
        
        return target_demo, main_category, subcategory, base_price
    
    for i in range(n_samples):
        # Select category and get base pricing
        target_demo, main_category, subcategory, base_price = select_category_and_pricing()
        
        # Product level features
        brand = np.random.choice(brands)
        
        # Adjust pricing based on brand
        brand_multiplier = {'Local Boutique': 0.7, 'Unbranded': 0.6, 'Mid-tier Brand': 1.0, 'Premium Brand': 1.8}
        mrp_price = base_price * brand_multiplier.get(brand, 1.0)
        
        # Size selection based on demographic
        if target_demo == 'kids':
            size = np.random.choice(['0-3M', '3-6M', '6-12M', '1-2Y', '2-3Y', '3-4Y', '4-5Y', '6-7Y', '8-9Y', '10-11Y', '12-13Y', '14-15Y'])
        else:
            size = np.random.choice(['S', 'M', 'L', 'XL', 'XXL', 'Free Size'])
            
        color = np.random.choice(colors)
        discount_percent = np.random.uniform(10, 70)
        discounted_price = mrp_price * (1 - discount_percent/100)
        
        # Material selection based on category
        if main_category in ['Sarees', 'Lehengas', 'Suits & Dress Material']:
            material = np.random.choice(['silk', 'cotton', 'georgette', 'chiffon', 'net', 'blend'])
        elif target_demo == 'kids':
            material = np.random.choice(['cotton', 'blend', 'polyester'])  # Softer materials for kids
        else:
            material = np.random.choice(materials)
            
        fit_type = np.random.choice(fit_types)
        seller_rating = np.random.uniform(3.0, 5.0)
        stock_type = np.random.choice(['fast-fashion', 'evergreen'])
        images_quality_score = np.random.uniform(1, 10)
        
        # Basket level features
        basket_size = np.random.randint(1, 8)
        basket_diversity = np.random.randint(1, min(basket_size + 1, 5))
        basket_value = discounted_price * basket_size * np.random.uniform(0.8, 1.2)
        channel = np.random.choice(channels)
        device_type = np.random.choice(device_types)
        os_version = np.random.uniform(8.0, 15.0)
        payment_method = np.random.choice(payment_methods)
        order_hour = np.random.randint(0, 24)
        delivery_option = np.random.choice(['standard', 'express'])
        
        # Customer level features
        past_return_rate = np.random.uniform(0, 0.4)
        past_purchase_volume = np.random.randint(1, 50)
        category_affinity = np.random.choice(['kurta_heavy', 'western_heavy', 'mixed', 'saree_heavy'])
        primary_return_reason = np.random.choice(past_return_reasons)
        city_tier = np.random.choice(city_tiers)
        avg_basket_value = np.random.uniform(500, 3000)
        payment_preference = np.random.choice(['COD_heavy', 'prepaid_heavy', 'mixed'])
        fraud_score = np.random.uniform(0, 1)
        tenure_days = np.random.randint(30, 1095)  # 30 days to 3 years
        
        # Target variable with realistic return rate (25-35%)
        # Base return probability factors
        base_return_prob = 0.20  # Start with 20% base
        
        # Category-specific return rates (adjusted for 25-33% overall target)
        category_return_rates = {
            'women_ethnic': {
                'Sarees': 0.23, 'Kurtis': 0.16, 'Kurta Sets': 0.18, 'Suits & Dress Material': 0.20,
                'Lehengas': 0.26, 'Dupatta Sets': 0.14, 'Other Ethnic': 0.19
            },
            'women_western': {
                'Dresses': 0.23, 'Jeans': 0.26, 'Tops': 0.16, 'T-shirts': 0.12, 'Shirts': 0.14,
                'Trousers': 0.23, 'Shorts': 0.18, 'Skirts': 0.20, 'Jumpsuits': 0.24, 'Co-ord Sets': 0.19
            },
            'men': {
                'T-shirts': 0.11, 'Shirts': 0.14, 'Jeans': 0.23, 'Trousers': 0.19, 'Shorts': 0.16,
                'Ethnic Wear': 0.18, 'Jackets': 0.20, 'Sweatshirts': 0.12, 'Track Suits': 0.09, 'Innerwear': 0.07
            },
            'kids': {
                'Boys Clothing': 0.12, 'Girls Clothing': 0.14, 'Ethnic Wear': 0.16,
                'Footwear': 0.19, 'Accessories': 0.09, 'Baby Care': 0.06
            }
        }
        
        # Get category-specific return rate
        category_base_rate = category_return_rates.get(target_demo, {}).get(main_category, 0.3)
        
        # Adjust based on various factors
        return_probability = category_base_rate
        
        # Size-related returns (major factor in fashion)
        if size in ['S', 'XS', 'XXL', 'XXXL']:
            return_probability += 0.05  # Extreme sizes have higher return rates
        elif size in ['Free Size']:
            return_probability += 0.03  # Free size can be inconsistent
        
        # Discount impact (higher discount = impulse buying = more returns)
        if discount_percent > 60:
            return_probability += 0.08
        elif discount_percent > 40:
            return_probability += 0.05
        elif discount_percent > 20:
            return_probability += 0.02
        
        # Payment method impact
        if payment_method == 'COD':
            return_probability += 0.06  # COD has higher return rates
        elif payment_method == 'prepaid':
            return_probability -= 0.03  # Prepaid customers more committed
        
        # Customer history impact
        return_probability += past_return_rate * 0.2  # Past behavior predicts future
        
        # Price impact (very cheap items might have quality issues)
        if discounted_price < 200:
            return_probability += 0.05
        elif discounted_price > 2000:
            return_probability -= 0.02  # Higher price = better quality expectations met
        
        # Seller rating impact
        if seller_rating < 3.5:
            return_probability += 0.06
        elif seller_rating > 4.5:
            return_probability -= 0.03
        
        # Fast fashion tends to have higher returns
        if stock_type == 'fast-fashion':
            return_probability += 0.04
        
        # Order timing (late night impulse orders)
        if order_hour >= 22 or order_hour <= 6:
            return_probability += 0.03
        
        # City tier impact
        if city_tier == 'tier-3':
            return_probability += 0.02  # Less familiar with online shopping
        elif city_tier == 'metro':
            return_probability -= 0.01
        
        # Image quality impact
        if images_quality_score < 4:
            return_probability += 0.05  # Poor images lead to wrong expectations
        elif images_quality_score > 8:
            return_probability -= 0.02
        
        # Fraud indicators
        return_probability += fraud_score * 0.08
        
        # Add some randomness but keep it controlled
        return_probability += np.random.normal(0, 0.03)
        
        # Ensure probability stays in realistic range for 25-33% target
        return_probability = max(0.05, min(0.45, return_probability))  # Keep between 5% and 45%
        
        # Final binary decision - adjusted to achieve 25-33% return rate
        returned = 1 if np.random.random() < return_probability else 0
        
        data.append({
            # Product Level Features
            'target_demographic': target_demo,
            'brand_label': brand,
            'main_category': main_category,
            'subcategory': subcategory,
            'size': size,
            'color': color,
            'price_mrp': round(mrp_price, 2),
            'price_discounted': round(discounted_price, 2),
            'discount_percent': round(discount_percent, 1),
            'material_fabric': material,
            'fit_type': fit_type,
            'seller_rating': round(seller_rating, 1),
            'stock_type': stock_type,
            'images_quality_score': round(images_quality_score, 1),
            
            # Basket Level Features
            'basket_size': basket_size,
            'basket_diversity': basket_diversity,
            'basket_value': round(basket_value, 2),
            'channel': channel,
            'device_type': device_type,
            'os_version': round(os_version, 1),
            'payment_method': payment_method,
            'order_hour': order_hour,
            'delivery_option': delivery_option,
            
            # Customer Level Features
            'past_return_rate': round(past_return_rate, 3),
            'past_purchase_volume': past_purchase_volume,
            'category_affinity': category_affinity,
            'primary_return_reason': primary_return_reason,
            'city_tier': city_tier,
            'avg_basket_value': round(avg_basket_value, 2),
            'payment_preference': payment_preference,
            'fraud_score': round(fraud_score, 3),
            'tenure_days': tenure_days,
            
            # Target
            'returned': returned,
            'return_probability': round(return_probability, 3)
        })
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_sample_data(1000)

# Create train/validation/test splits
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print("Fashion E-commerce Features Dataset Created!")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Total features: {len(df.columns)}")

# Display feature categories
print("\n=== FEATURE CATEGORIES ===")
for category, info in feature_categories.items():
    print(f"\n{category.upper().replace('_', ' ')} FEATURES:")
    print(f"Description: {info['description']}")
    for feature, description in info['features'].items():
        print(f"  • {feature}: {description}")

# Display sample data
print("\n=== SAMPLE DATA ===")
print(df.head())

# Display data info
print("\n=== DATA INFO ===")
print(df.info())

# Display pricing analysis by demographic
print("\n=== PRICING ANALYSIS BY DEMOGRAPHIC ===")
pricing_stats = df.groupby('target_demographic').agg({
    'price_mrp': ['mean', 'min', 'max'],
    'price_discounted': ['mean', 'min', 'max'],
    'discount_percent': 'mean'
}).round(2)

print(pricing_stats)

# Display category distribution
print("\n=== CATEGORY DISTRIBUTION ===")
print(df.groupby(['target_demographic', 'main_category']).size().head(15))

# Display basic statistics
print("\n=== BASIC STATISTICS ===")
print(f"Return rate: {df['returned'].mean():.2%}")
print(f"Average basket value: ₹{df['basket_value'].mean():.2f}")
print(f"Average discount: {df['discount_percent'].mean():.1f}%")
print(f"Kids items average price: ₹{df[df['target_demographic'] == 'kids']['price_discounted'].mean():.2f}")
print(f"Adult items average price: ₹{df[df['target_demographic'] != 'kids']['price_discounted'].mean():.2f}")

# Save as CSV
df.to_csv('fashion_ecommerce_features.csv', index=False)
print("\n✅ Dataset saved as 'fashion_ecommerce_features.csv'")

# Feature importance insights based on the original framework
feature_insights = {
    "high_return_risk_indicators": [
        "High discount percentage (>50%)",
        "COD payment method", 
        "Late night orders (22:00-06:00)",
        "High past return rate",
        "Low seller ratings",
        "Fast-fashion stock type"
    ],
    "low_return_risk_indicators": [
        "Prepaid/UPI payments",
        "High tenure customers", 
        "Premium brand items",
        "Good image quality scores",
        "Metro city customers",
        "Daytime orders"
    ]
}

print("\n=== KEY INSIGHTS FROM FRAMEWORK ===")
for category, indicators in feature_insights.items():
    print(f"\n{category.upper().replace('_', ' ')}:")
    for indicator in indicators:
        print(f"  • {indicator}")