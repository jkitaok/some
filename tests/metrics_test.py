"""
Unit tests for extraction/metrics.py module.

Tests metrics collection and reporting functionality.
"""
import unittest
import tempfile
import os
import json
import time
from typing import List, Optional, Dict, Any
from unittest.mock import patch, MagicMock

try:
    from pydantic import BaseModel
    from some.metrics import SchemaMetricsCollector
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    SchemaMetricsCollector = None


class TestBaseMetricsCollector(unittest.TestCase):
    """Test cases for BaseMetricsCollector abstract class."""

    def test_base_metrics_collector_is_abstract(self):
        """Test that BaseMetricsCollector cannot be instantiated directly."""
        from some.metrics import BaseMetricsCollector
        
        with self.assertRaises(TypeError):
            BaseMetricsCollector()

    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined."""
        from some.metrics import BaseMetricsCollector
        
        # Create a concrete subclass without implementing abstract methods
        class IncompleteCollector(BaseMetricsCollector):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteCollector()


class TestLLMMetricsCollector(unittest.TestCase):
    """Test cases for LLMMetricsCollector."""

    def setUp(self):
        """Set up test fixtures."""
        from some.metrics import LLMMetricsCollector
        self.collector = LLMMetricsCollector(
            name="test_extraction",
            cost_per_input_token=0.001,
            cost_per_output_token=0.002
        )

    def test_initialization(self):
        """Test LLMMetricsCollector initialization."""
        self.assertEqual(self.collector.name, "test_extraction")
        self.assertEqual(self.collector.cost_per_input_token, 0.001)
        self.assertEqual(self.collector.cost_per_output_token, 0.002)

    def test_no_manual_timing(self):
        """Test that manual timing methods are not available."""
        # LLMMetricsCollector should rely on timing from language model results
        self.assertFalse(hasattr(self.collector, 'start_timing'))
        self.assertFalse(hasattr(self.collector, 'end_timing'))
        self.assertFalse(hasattr(self.collector, 'get_total_time'))

    def test_collect_metrics_empty_results(self):
        """Test collect_metrics with empty results."""
        metrics = self.collector.collect_metrics([])
        
        expected = {
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0
        }
        self.assertEqual(metrics, expected)

    def test_collect_metrics_with_results(self):
        """Test collect_metrics with sample results."""
        results = [
            {"input_tokens": 100, "output_tokens": 50, "result": "success"},
            {"input_tokens": 150, "output_tokens": 75, "result": "success"},
            {"input_tokens": 80, "output_tokens": 0, "error": "failed", "result": None}
        ]

        # Add inference time from language model generate() call
        self.collector.add_inference_time(0.9)  # Total time for all calls
        
        metrics = self.collector.collect_metrics(results)
        
        self.assertEqual(metrics["total_items"], 3)
        self.assertEqual(metrics["successful_items"], 2)
        self.assertEqual(metrics["failed_items"], 1)
        self.assertEqual(metrics["total_input_tokens"], 330)
        self.assertEqual(metrics["total_output_tokens"], 125)
        self.assertEqual(metrics["avg_input_tokens"], 110.0)
        self.assertAlmostEqual(metrics["avg_output_tokens"], 41.67, places=2)
        
        # Cost calculation: (330 * 0.001) + (125 * 0.002) = 0.33 + 0.25 = 0.58
        self.assertAlmostEqual(metrics["total_cost"], 0.58, places=3)

        # Timing metrics
        self.assertAlmostEqual(metrics["total_inference_time"], 0.9, places=3)  # Total time added
        self.assertAlmostEqual(metrics["avg_processing_time"], 0.3, places=3)   # 0.9 / 3 results

    def test_format_summary(self):
        """Test format_summary method."""
        metrics = {
            "extraction_name": "test_extraction",
            "total_items": 10,
            "successful_items": 8,
            "failed_items": 2,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "avg_input_tokens": 100.0,
            "avg_output_tokens": 50.0,
            "total_cost": 1.5,
            "success_rate": 0.8,
            "total_time": 0.0,
            "avg_processing_time": 0.0,
            "total_tokens": 1500,
            "errors": []
        }
        
        summary = self.collector.format_summary(metrics)
        
        self.assertIn("Test_Extraction", summary)
        self.assertIn("Total items: 10", summary)
        self.assertIn("Successful: 8", summary)
        self.assertIn("Failed: 2", summary)
        self.assertIn("Input: 1,000", summary)
        self.assertIn("Output: 500", summary)
        self.assertIn("$1.50", summary)


class TestCreateMetricsCollector(unittest.TestCase):
    """Test cases for create_metrics_collector factory function."""

    def test_create_standard_collector(self):
        """Test creating standard metrics collector."""
        from some.metrics import create_metrics_collector, LLMMetricsCollector
        
        collector = create_metrics_collector(
            collector_type="standard",
            name="test",
            cost_per_input_token=0.001,
            cost_per_output_token=0.002
        )
        
        self.assertIsInstance(collector, LLMMetricsCollector)
        self.assertEqual(collector.name, "test")
        self.assertEqual(collector.cost_per_input_token, 0.001)
        self.assertEqual(collector.cost_per_output_token, 0.002)

    def test_create_unknown_collector_raises_error(self):
        """Test creating unknown collector type raises ValueError."""
        from some.metrics import create_metrics_collector
        
        with self.assertRaises(ValueError) as context:
            create_metrics_collector(collector_type="unknown")
        
        self.assertIn("Unknown collector", str(context.exception))

    def test_create_collector_default_values(self):
        """Test creating collector with default values."""
        from some.metrics import create_metrics_collector, LLMMetricsCollector

        collector = create_metrics_collector()

        self.assertIsInstance(collector, LLMMetricsCollector)
        self.assertEqual(collector.name, "extraction")
        self.assertEqual(collector.cost_per_input_token, 0.0)
        self.assertEqual(collector.cost_per_output_token, 0.0)


class TestSaveMetricsJson(unittest.TestCase):
    """Test cases for save_metrics_json function."""

    def test_save_metrics_json(self):
        """Test saving metrics to JSON file."""
        from some.metrics import save_metrics_json
        from some.io import read_json
        
        test_metrics = {
            "total_items": 5,
            "successful_items": 4,
            "failed_items": 1,
            "total_cost": 2.5
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_metrics_json(test_metrics, temp_path)
            
            # Verify file was written correctly
            result = read_json(temp_path)
            self.assertEqual(result, test_metrics)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_metrics_json_creates_directories(self):
        """Test that save_metrics_json creates parent directories."""
        from some.metrics import save_metrics_json
        from some.io import read_json
        
        test_metrics = {"test": "data"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "subdir", "metrics.json")
            
            save_metrics_json(test_metrics, nested_path)
            
            # Verify file was written and directories created
            self.assertTrue(os.path.exists(nested_path))
            result = read_json(nested_path)
            self.assertEqual(result, test_metrics)


@unittest.skipUnless(PYDANTIC_AVAILABLE, "Pydantic not available")
class TestSchemaMetricsCollector(unittest.TestCase):
    """Test cases for SchemaMetricsCollector."""

    def setUp(self):
        """Set up test schemas."""
        class TestProduct(BaseModel):
            name: str
            price: float
            in_stock: bool
            features: List[str]
            description: Optional[str] = None

        class TestUser(BaseModel):
            username: str
            age: int
            active: bool

        class TestNestedSchema(BaseModel):
            title: str
            product: TestProduct
            user: TestUser

        self.TestProduct = TestProduct
        self.TestUser = TestUser
        self.TestNestedSchema = TestNestedSchema

    def test_schema_metrics_collector_init(self):
        """Test SchemaMetricsCollector initialization."""
        collector = SchemaMetricsCollector(self.TestProduct, "test_products")

        self.assertEqual(collector.name, "test_products")
        self.assertEqual(collector.schema_class, self.TestProduct)
        self.assertEqual(collector.schema_name, "TestProduct")

    def test_schema_metrics_collector_init_invalid_schema(self):
        """Test SchemaMetricsCollector with invalid schema."""
        with self.assertRaises(ValueError):
            SchemaMetricsCollector(str, "test")

    def test_collect_metrics_empty_list(self):
        """Test collecting metrics from empty list."""
        collector = SchemaMetricsCollector(self.TestProduct)
        metrics = collector.collect_metrics([])

        self.assertEqual(metrics["schema_name"], "TestProduct")
        self.assertEqual(metrics["total_objects"], 0)
        self.assertEqual(metrics["field_metrics"], {})

    def test_collect_metrics_numeric_fields(self):
        """Test collecting metrics for numeric fields."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "Product A", "price": 10.99, "in_stock": True, "features": ["feature1"], "description": "desc1"},
            {"name": "Product B", "price": 25.50, "in_stock": False, "features": ["feature2"], "description": None},
            {"name": "Product C", "price": 15.75, "in_stock": True, "features": [], "description": "desc3"},
        ]

        metrics = collector.collect_metrics(test_data)

        price_metrics = metrics["field_metrics"]["price"]
        self.assertEqual(price_metrics["type"], "float")
        self.assertEqual(price_metrics["total_count"], 3)
        self.assertEqual(price_metrics["non_none_count"], 3)
        self.assertEqual(price_metrics["none_count"], 0)
        self.assertAlmostEqual(price_metrics["mean"], 17.41, places=2)
        self.assertEqual(price_metrics["min"], 10.99)
        self.assertEqual(price_metrics["max"], 25.50)

    def test_collect_metrics_boolean_fields(self):
        """Test collecting metrics for boolean fields."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "A", "price": 10.0, "in_stock": True, "features": [], "description": None},
            {"name": "B", "price": 20.0, "in_stock": False, "features": [], "description": None},
            {"name": "C", "price": 30.0, "in_stock": True, "features": [], "description": None},
            {"name": "D", "price": 40.0, "in_stock": True, "features": [], "description": None},
        ]

        metrics = collector.collect_metrics(test_data)

        stock_metrics = metrics["field_metrics"]["in_stock"]
        self.assertEqual(stock_metrics["type"], "bool")
        self.assertEqual(stock_metrics["true_count"], 3)
        self.assertEqual(stock_metrics["false_count"], 1)
        self.assertEqual(stock_metrics["true_percentage"], 75.0)
        self.assertEqual(stock_metrics["false_percentage"], 25.0)

    def test_collect_metrics_string_fields(self):
        """Test collecting metrics for string fields."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "Short", "price": 10.0, "in_stock": True, "features": [], "description": "A"},
            {"name": "Medium Name", "price": 20.0, "in_stock": False, "features": [], "description": "Longer description"},
            {"name": "Very Long Product Name", "price": 30.0, "in_stock": True, "features": [], "description": None},
        ]

        metrics = collector.collect_metrics(test_data)

        name_metrics = metrics["field_metrics"]["name"]
        self.assertEqual(name_metrics["type"], "str")
        self.assertEqual(name_metrics["total_count"], 3)
        self.assertEqual(name_metrics["non_none_count"], 3)
        self.assertEqual(name_metrics["min_length"], 5)  # "Short"
        self.assertEqual(name_metrics["max_length"], 22)  # "Very Long Product Name"

    def test_collect_metrics_list_fields(self):
        """Test collecting metrics for list fields."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "A", "price": 10.0, "in_stock": True, "features": ["f1", "f2", "f3"], "description": None},
            {"name": "B", "price": 20.0, "in_stock": False, "features": [], "description": None},
            {"name": "C", "price": 30.0, "in_stock": True, "features": ["feature1"], "description": None},
        ]

        metrics = collector.collect_metrics(test_data)

        features_metrics = metrics["field_metrics"]["features"]
        self.assertEqual(features_metrics["type"], "list[str]")
        self.assertEqual(features_metrics["total_count"], 3)
        self.assertEqual(features_metrics["empty_list_count"], 1)
        self.assertEqual(features_metrics["min_length"], 0)
        self.assertEqual(features_metrics["max_length"], 3)
        self.assertAlmostEqual(features_metrics["avg_length"], 1.33, places=2)

    def test_collect_metrics_optional_fields(self):
        """Test collecting metrics for optional fields."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "A", "price": 10.0, "in_stock": True, "features": [], "description": "Has description"},
            {"name": "B", "price": 20.0, "in_stock": False, "features": [], "description": None},
            {"name": "C", "price": 30.0, "in_stock": True, "features": [], "description": "Another description"},
        ]

        metrics = collector.collect_metrics(test_data)

        desc_metrics = metrics["field_metrics"]["description"]
        self.assertEqual(desc_metrics["type"], "str")
        self.assertEqual(desc_metrics["total_count"], 3)
        self.assertEqual(desc_metrics["non_none_count"], 2)
        self.assertEqual(desc_metrics["none_count"], 1)
        self.assertTrue(desc_metrics["is_optional"])

    def test_collect_metrics_all_none_values(self):
        """Test collecting metrics when all values are None."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "A", "price": 10.0, "in_stock": True, "features": [], "description": None},
            {"name": "B", "price": 20.0, "in_stock": False, "features": [], "description": None},
        ]

        metrics = collector.collect_metrics(test_data)

        desc_metrics = metrics["field_metrics"]["description"]
        self.assertEqual(desc_metrics["none_count"], 2)
        self.assertEqual(desc_metrics["non_none_count"], 0)
        # Should only have base metrics when all values are None
        self.assertNotIn("avg_length", desc_metrics)

    def test_collect_metrics_invalid_data_types(self):
        """Test collecting metrics with invalid data types."""
        collector = SchemaMetricsCollector(self.TestProduct)

        test_data = [
            {"name": "A", "price": "invalid_price", "in_stock": True, "features": [], "description": None},
            {"name": "B", "price": 20.0, "in_stock": "invalid_bool", "features": [], "description": None},
        ]

        metrics = collector.collect_metrics(test_data)

        price_metrics = metrics["field_metrics"]["price"]
        self.assertEqual(price_metrics["valid_numeric_count"], 1)
        self.assertEqual(price_metrics["invalid_values"], 1)

        stock_metrics = metrics["field_metrics"]["in_stock"]
        self.assertEqual(stock_metrics["valid_boolean_count"], 1)
        self.assertEqual(stock_metrics["invalid_values"], 1)

    def test_format_summary(self):
        """Test formatting metrics summary."""
        collector = SchemaMetricsCollector(self.TestProduct, "test_products")

        test_data = [
            {"name": "Product A", "price": 10.99, "in_stock": True, "features": ["f1"], "description": "desc"},
            {"name": "Product B", "price": 25.50, "in_stock": False, "features": [], "description": None},
        ]

        metrics = collector.collect_metrics(test_data)
        summary = collector.format_summary(metrics)

        self.assertIn("TestProduct Schema Analysis", summary)
        self.assertIn("Total objects analyzed: 2", summary)
        self.assertIn("name", summary)
        self.assertIn("(str)", summary)
        self.assertIn("price", summary)
        self.assertIn("(float)", summary)
        self.assertIn("Mean:", summary)
        self.assertIn("True:", summary)

    def test_nested_object_analysis(self):
        """Test analysis of nested objects."""
        collector = SchemaMetricsCollector(self.TestNestedSchema)

        test_data = [
            {
                "title": "Test 1",
                "product": {"name": "P1", "price": 10.0, "in_stock": True, "features": [], "description": None},
                "user": {"username": "user1", "age": 25, "active": True}
            },
            {
                "title": "Test 2",
                "product": {"name": "P2", "price": 20.0, "in_stock": False, "features": ["f1"], "description": "desc"},
                "user": {"username": "user2", "age": 30, "active": False}
            }
        ]

        metrics = collector.collect_metrics(test_data)

        product_metrics = metrics["field_metrics"]["product"]
        self.assertEqual(product_metrics["nested_schema"], "TestProduct")
        self.assertEqual(product_metrics["valid_object_count"], 2)
        self.assertIn("nested_metrics", product_metrics)

        # Check that nested metrics contain expected fields
        nested_metrics = product_metrics["nested_metrics"]
        self.assertIn("name", nested_metrics)
        self.assertIn("price", nested_metrics)

    def test_deep_nested_collection_analysis(self):
        """Test deep analysis of nested collections."""
        class ComplexSchema(BaseModel):
            data: List[Dict[str, List[int]]]
            metadata: Dict[str, List[str]]

        collector = SchemaMetricsCollector(ComplexSchema)

        test_data = [
            {
                "data": [
                    {"numbers": [1, 2, 3], "values": [10, 20]},
                    {"numbers": [4, 5], "values": [30, 40, 50]}
                ],
                "metadata": {
                    "tags": ["tag1", "tag2"],
                    "categories": ["cat1", "cat2", "cat3"]
                }
            },
            {
                "data": [
                    {"numbers": [6, 7, 8, 9], "values": [60]}
                ],
                "metadata": {
                    "tags": ["tag3"],
                    "categories": ["cat4"]
                }
            }
        ]

        metrics = collector.collect_metrics(test_data)

        # Check that deep nested analysis was performed
        data_metrics = metrics["field_metrics"]["data"]
        self.assertIn("item_metrics", data_metrics)

        metadata_metrics = metrics["field_metrics"]["metadata"]
        self.assertIn("key_value_metrics", metadata_metrics)

        # Verify deep analysis includes type distribution
        item_metrics = data_metrics["item_metrics"]
        self.assertIn("type_analyses", item_metrics)

    def test_mixed_type_collection_analysis(self):
        """Test analysis of collections with mixed data types."""
        class MixedSchema(BaseModel):
            mixed_list: List[Any]

        collector = SchemaMetricsCollector(MixedSchema)

        test_data = [
            {
                "mixed_list": [
                    "string1",
                    42,
                    True,
                    [1, 2, 3],
                    {"nested": "dict"},
                    3.14,
                    False,
                    "string2"
                ]
            },
            {
                "mixed_list": [
                    "string3",
                    100,
                    [4, 5],
                    {"another": "dict", "with": "more", "keys": "here"}
                ]
            }
        ]

        metrics = collector.collect_metrics(test_data)

        # Check mixed type analysis
        mixed_metrics = metrics["field_metrics"]["mixed_list"]
        self.assertIn("item_metrics", mixed_metrics)

        item_metrics = mixed_metrics["item_metrics"]
        self.assertIn("type_distribution", item_metrics)
        self.assertIn("type_analyses", item_metrics)

        # Verify different types were detected and analyzed
        type_analyses = item_metrics["type_analyses"]
        self.assertIn("str", type_analyses)
        self.assertIn("int", type_analyses)
        self.assertIn("bool", type_analyses)
        self.assertIn("list", type_analyses)
        self.assertIn("dict", type_analyses)


class TestCreateMetricsCollectorUpdated(unittest.TestCase):
    """Test cases for updated create_metrics_collector factory function."""

    def test_create_schema_collector(self):
        """Test creating schema metrics collector."""
        if not PYDANTIC_AVAILABLE:
            self.skipTest("Pydantic not available")

        from some.metrics import create_metrics_collector

        class TestSchema(BaseModel):
            name: str
            value: int

        collector = create_metrics_collector(
            collector_type="schema",
            name="test_schema",
            schema_class=TestSchema
        )

        self.assertIsInstance(collector, SchemaMetricsCollector)
        self.assertEqual(collector.name, "test_schema")
        self.assertEqual(collector.schema_class, TestSchema)

    def test_create_schema_collector_missing_schema(self):
        """Test creating schema collector without schema_class."""
        if not PYDANTIC_AVAILABLE:
            self.skipTest("Pydantic not available")

        from some.metrics import create_metrics_collector

        with self.assertRaises(ValueError) as context:
            create_metrics_collector(collector_type="schema")

        self.assertIn("schema_class is required", str(context.exception))


if __name__ == '__main__':
    unittest.main()
