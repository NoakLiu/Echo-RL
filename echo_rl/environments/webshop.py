"""
WebShop Environment Interface

Implements the WebShop environment for web-based shopping agent tasks.
WebShop provides realistic e-commerce scenarios where agents must navigate
websites, search for products, and complete purchases.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
import re

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentState

logger = logging.getLogger(__name__)

@dataclass
class WebShopConfig(EnvironmentConfig):
    """Configuration for WebShop environment"""
    website_type: str = "electronics"  # electronics, clothing, books, etc.
    max_search_results: int = 20
    max_cart_items: int = 10
    budget_limit: float = 1000.0
    state_encoding_dim: int = 512
    action_space_size: int = 25  # WebShop actions

class WebShopEnvironment(EchoRLEnvironment):
    """
    WebShop environment for web-based shopping tasks
    
    Simulates realistic e-commerce interactions including product search,
    comparison, and purchase completion.
    """
    
    def __init__(self, config: WebShopConfig):
        super().__init__(config)
        self.config = config
        
        # Environment state
        self.current_page = "home"
        self.search_query = ""
        self.search_results = []
        self.product_details = {}
        self.shopping_cart = []
        self.total_cost = 0.0
        self.user_preferences = {}
        self.task_requirements = {}
        
        # Action mapping
        self.action_map = self._create_action_map()
        
        # Product database (simplified)
        self.product_database = self._create_product_database()
        
        # Task setup
        self._setup_shopping_task()
    
    def _create_action_map(self) -> Dict[int, str]:
        """Create mapping from action indices to WebShop commands"""
        return {
            0: "search",
            1: "click_product",
            2: "add_to_cart",
            3: "remove_from_cart",
            4: "view_cart",
            5: "checkout",
            6: "filter_price",
            7: "filter_rating",
            8: "sort_by_price",
            9: "sort_by_rating",
            10: "sort_by_relevance",
            11: "view_details",
            12: "compare_products",
            13: "read_reviews",
            14: "navigate_home",
            15: "navigate_category",
            16: "navigate_deals",
            17: "navigate_account",
            18: "apply_coupon",
            19: "change_quantity",
            20: "select_shipping",
            21: "enter_payment",
            22: "confirm_order",
            23: "back",
            24: "refresh"
        }
    
    def _create_product_database(self) -> Dict[str, List[Dict]]:
        """Create simplified product database"""
        if self.config.website_type == "electronics":
            return {
                "laptops": [
                    {"id": "laptop1", "name": "MacBook Pro", "price": 1299.99, "rating": 4.8, "category": "laptops"},
                    {"id": "laptop2", "name": "Dell XPS 13", "price": 999.99, "rating": 4.6, "category": "laptops"},
                    {"id": "laptop3", "name": "HP Spectre", "price": 1199.99, "rating": 4.4, "category": "laptops"}
                ],
                "phones": [
                    {"id": "phone1", "name": "iPhone 15", "price": 799.99, "rating": 4.9, "category": "phones"},
                    {"id": "phone2", "name": "Samsung Galaxy S24", "price": 699.99, "rating": 4.7, "category": "phones"},
                    {"id": "phone3", "name": "Google Pixel 8", "price": 599.99, "rating": 4.5, "category": "phones"}
                ],
                "accessories": [
                    {"id": "acc1", "name": "Wireless Headphones", "price": 199.99, "rating": 4.3, "category": "accessories"},
                    {"id": "acc2", "name": "Phone Case", "price": 29.99, "rating": 4.2, "category": "accessories"},
                    {"id": "acc3", "name": "Laptop Stand", "price": 49.99, "rating": 4.1, "category": "accessories"}
                ]
            }
        else:
            # Default product database
            return {
                "products": [
                    {"id": "prod1", "name": "Product 1", "price": 99.99, "rating": 4.0, "category": "general"},
                    {"id": "prod2", "name": "Product 2", "price": 149.99, "rating": 4.2, "category": "general"},
                    {"id": "prod3", "name": "Product 3", "price": 199.99, "rating": 4.5, "category": "general"}
                ]
            }
    
    def _setup_shopping_task(self):
        """Setup shopping task requirements"""
        # Random task: find and purchase specific products
        self.task_requirements = {
            "target_product": "laptop",
            "max_price": 1200.0,
            "min_rating": 4.5,
            "required_features": ["portable", "fast"],
            "budget": self.config.budget_limit
        }
    
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset shopping state
        self.current_page = "home"
        self.search_query = ""
        self.search_results = []
        self.product_details = {}
        self.shopping_cart = []
        self.total_cost = 0.0
        
        # Generate initial observation
        observation = self._generate_observation()
        
        # Create initial state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=0.0,
            done=False,
            info={
                "current_page": self.current_page,
                "search_results_count": len(self.search_results),
                "cart_items": len(self.shopping_cart),
                "total_cost": self.total_cost,
                "task_requirements": self.task_requirements
            },
            step_count=self.current_step,
            episode_reward=self.episode_reward
        )
        
        return state
    
    def step(self, action: int) -> EnvironmentState:
        """Execute action and return next state"""
        self.current_step += 1
        
        # Get action command
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        if action not in self.action_map:
            action = 24  # Default to refresh
        
        action_command = self.action_map[action]
        
        # Execute action and get reward
        reward, success = self._execute_action(action_command)
        self.episode_reward += reward
        
        # Check if task is complete
        done = self._is_task_complete() or self.is_done()
        
        # Generate new observation
        observation = self._generate_observation()
        
        # Create next state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=reward,
            done=done,
            info={
                "action_executed": action_command,
                "success": success,
                "current_page": self.current_page,
                "search_results_count": len(self.search_results),
                "cart_items": len(self.shopping_cart),
                "total_cost": self.total_cost,
                "task_progress": self._get_task_progress()
            },
            step_count=self.current_step,
            episode_reward=self.episode_reward
        )
        
        # Update statistics
        if done:
            self._update_statistics(reward, success)
            self._finish_episode()
        
        return state
    
    def _execute_action(self, action_command: str) -> Tuple[float, bool]:
        """Execute action command and return reward and success flag"""
        reward = 0.0
        success = False
        
        if action_command == "search":
            reward, success = self._action_search()
        elif action_command == "click_product":
            reward, success = self._action_click_product()
        elif action_command == "add_to_cart":
            reward, success = self._action_add_to_cart()
        elif action_command == "remove_from_cart":
            reward, success = self._action_remove_from_cart()
        elif action_command == "view_cart":
            reward, success = self._action_view_cart()
        elif action_command == "checkout":
            reward, success = self._action_checkout()
        elif action_command == "filter_price":
            reward, success = self._action_filter_price()
        elif action_command == "filter_rating":
            reward, success = self._action_filter_rating()
        elif action_command == "sort_by_price":
            reward, success = self._action_sort_by_price()
        elif action_command == "sort_by_rating":
            reward, success = self._action_sort_by_rating()
        elif action_command == "navigate_home":
            reward, success = self._action_navigate_home()
        elif action_command == "navigate_category":
            reward, success = self._action_navigate_category()
        else:
            # Default action
            reward = 0.0
            success = True
        
        return reward, success
    
    def _action_search(self) -> Tuple[float, bool]:
        """Perform product search"""
        # Simulate search for target product
        target = self.task_requirements["target_product"]
        
        # Find matching products
        matching_products = []
        for category, products in self.product_database.items():
            for product in products:
                if target.lower() in product["name"].lower() or target.lower() in product["category"].lower():
                    matching_products.append(product)
        
        self.search_results = matching_products[:self.config.max_search_results]
        self.current_page = "search_results"
        
        if self.search_results:
            return 1.0, True
        else:
            return -0.1, False
    
    def _action_click_product(self) -> Tuple[float, bool]:
        """Click on a product to view details"""
        if not self.search_results:
            return -0.1, False
        
        # Select first product
        product = self.search_results[0]
        self.product_details = product
        self.current_page = "product_details"
        
        return 0.5, True
    
    def _action_add_to_cart(self) -> Tuple[float, bool]:
        """Add current product to cart"""
        if not self.product_details:
            return -0.1, False
        
        # Check if product meets requirements
        meets_requirements = self._check_product_requirements(self.product_details)
        
        if meets_requirements:
            self.shopping_cart.append(self.product_details)
            self.total_cost += self.product_details["price"]
            return 2.0, True
        else:
            return -0.5, False
    
    def _action_remove_from_cart(self) -> Tuple[float, bool]:
        """Remove product from cart"""
        if not self.shopping_cart:
            return -0.1, False
        
        product = self.shopping_cart.pop()
        self.total_cost -= product["price"]
        return 0.1, True
    
    def _action_view_cart(self) -> Tuple[float, bool]:
        """View shopping cart"""
        self.current_page = "cart"
        return 0.1, True
    
    def _action_checkout(self) -> Tuple[float, bool]:
        """Proceed to checkout"""
        if not self.shopping_cart:
            return -0.1, False
        
        # Check if cart meets task requirements
        if self._is_cart_valid():
            self.current_page = "checkout"
            return 1.0, True
        else:
            return -0.5, False
    
    def _action_filter_price(self) -> Tuple[float, bool]:
        """Filter products by price"""
        max_price = self.task_requirements["max_price"]
        filtered_results = [p for p in self.search_results if p["price"] <= max_price]
        self.search_results = filtered_results
        return 0.5, True
    
    def _action_filter_rating(self) -> Tuple[float, bool]:
        """Filter products by rating"""
        min_rating = self.task_requirements["min_rating"]
        filtered_results = [p for p in self.search_results if p["rating"] >= min_rating]
        self.search_results = filtered_results
        return 0.5, True
    
    def _action_sort_by_price(self) -> Tuple[float, bool]:
        """Sort products by price"""
        self.search_results.sort(key=lambda x: x["price"])
        return 0.2, True
    
    def _action_sort_by_rating(self) -> Tuple[float, bool]:
        """Sort products by rating"""
        self.search_results.sort(key=lambda x: x["rating"], reverse=True)
        return 0.2, True
    
    def _action_navigate_home(self) -> Tuple[float, bool]:
        """Navigate to home page"""
        self.current_page = "home"
        return 0.1, True
    
    def _action_navigate_category(self) -> Tuple[float, bool]:
        """Navigate to category page"""
        self.current_page = "category"
        return 0.1, True
    
    def _check_product_requirements(self, product: Dict) -> bool:
        """Check if product meets task requirements"""
        # Check price
        if product["price"] > self.task_requirements["max_price"]:
            return False
        
        # Check rating
        if product["rating"] < self.task_requirements["min_rating"]:
            return False
        
        return True
    
    def _is_cart_valid(self) -> bool:
        """Check if cart meets task requirements"""
        if not self.shopping_cart:
            return False
        
        # Check total cost
        if self.total_cost > self.task_requirements["budget"]:
            return False
        
        # Check if cart contains target product type
        target_type = self.task_requirements["target_product"]
        has_target = any(target_type.lower() in item["category"].lower() for item in self.shopping_cart)
        
        return has_target
    
    def _is_task_complete(self) -> bool:
        """Check if shopping task is complete"""
        return (len(self.shopping_cart) > 0 and 
                self.current_page == "checkout" and 
                self._is_cart_valid())
    
    def _get_task_progress(self) -> float:
        """Get task completion progress"""
        progress = 0.0
        
        # Search progress
        if self.search_results:
            progress += 0.3
        
        # Cart progress
        if self.shopping_cart:
            progress += 0.4
        
        # Checkout progress
        if self.current_page == "checkout":
            progress += 0.3
        
        return min(progress, 1.0)
    
    def _generate_observation(self) -> str:
        """Generate text observation of current state"""
        obs_parts = []
        
        # Current page
        obs_parts.append(f"Current page: {self.current_page}")
        
        # Search results
        if self.search_results:
            obs_parts.append(f"Found {len(self.search_results)} products")
            for i, product in enumerate(self.search_results[:3]):  # Show first 3
                obs_parts.append(f"{i+1}. {product['name']} - ${product['price']} (Rating: {product['rating']})")
        
        # Cart status
        if self.shopping_cart:
            obs_parts.append(f"Cart: {len(self.shopping_cart)} items, Total: ${self.total_cost:.2f}")
        else:
            obs_parts.append("Cart: empty")
        
        # Task requirements
        obs_parts.append(f"Task: Find {self.task_requirements['target_product']} under ${self.task_requirements['max_price']}")
        
        return "\n".join(obs_parts)
    
    def get_state_representation(self) -> torch.Tensor:
        """Get current state as tensor representation"""
        # Create feature vector from environment state
        features = []
        
        # Page features
        page_features = [0.0] * 5  # home, search_results, product_details, cart, checkout
        page_map = {"home": 0, "search_results": 1, "product_details": 2, "cart": 3, "checkout": 4}
        if self.current_page in page_map:
            page_features[page_map[self.current_page]] = 1.0
        features.extend(page_features)
        
        # Search results features
        features.append(len(self.search_results) / self.config.max_search_results)
        
        # Cart features
        features.append(len(self.shopping_cart) / self.config.max_cart_items)
        features.append(self.total_cost / self.config.budget_limit)
        
        # Product quality features (if viewing product)
        if self.product_details:
            features.append(self.product_details["price"] / 2000.0)  # Normalized price
            features.append(self.product_details["rating"] / 5.0)  # Normalized rating
        else:
            features.extend([0.0, 0.0])
        
        # Task progress
        features.append(self._get_task_progress())
        
        # Pad or truncate to fixed size
        target_size = self.config.state_encoding_dim
        while len(features) < target_size:
            features.append(0.0)
        features = features[:target_size]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return self.config.action_space_size
    
    def get_state_dim(self) -> int:
        """Get dimension of state representation"""
        return self.config.state_encoding_dim
    
    def get_shopping_info(self) -> Dict[str, Any]:
        """Get shopping-specific information"""
        return {
            "website_type": self.config.website_type,
            "current_page": self.current_page,
            "search_results": self.search_results,
            "shopping_cart": self.shopping_cart,
            "total_cost": self.total_cost,
            "task_requirements": self.task_requirements,
            "product_database": self.product_database
        }
    
    def render(self, mode: str = "text") -> str:
        """Render environment state as text"""
        if mode == "text":
            return self._generate_observation()
        else:
            return super().render(mode)
