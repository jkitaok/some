"""
Generic metrics utilities for structured data extraction.

Provides customizable metrics collection and reporting for any extraction workflow.
"""
from __future__ import annotations

import json
import statistics
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union, Type, get_origin, get_args

try:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    FieldInfo = None

# Simple ANSI color helpers
_COLOR = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
}

def _c(text: str, color: str) -> str:
    return f"{_COLOR.get(color, '')}{text}{_COLOR['reset']}"


class BaseMetricsCollector(ABC):
    """Base class for metrics collection and reporting."""

    def __init__(self, name: str = "extraction"):
        self.name = name

    @abstractmethod
    def collect_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect metrics from extraction results."""
        pass

    @abstractmethod
    def format_summary(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a human-readable summary."""
        pass


class LLMMetricsCollector(BaseMetricsCollector):
    """LLM-focused metrics collector for token usage and processing times."""

    def __init__(self, name: str = "extraction", cost_per_input_token: float = 0.0, cost_per_output_token: float = 0.0):
        super().__init__(name)
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.total_inference_time = 0.0

    def add_inference_time(self, inference_time: float) -> None:
        """Add inference time from a language model generate() call."""
        self.total_inference_time += inference_time


    def collect_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect standard metrics from extraction results."""
        if not results:
            return {"total_items": 0, "successful_items": 0, "failed_items": 0}

        successful = [r for r in results if not r.get("error")]
        failed = [r for r in results if r.get("error")]

        # Token metrics
        total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
        avg_input_tokens = total_input_tokens / len(results) if results else 0
        avg_output_tokens = total_output_tokens / len(results) if results else 0

        # Cost calculation
        total_cost = (total_input_tokens * self.cost_per_input_token +
                     total_output_tokens * self.cost_per_output_token)

        # Use stored inference time from language model generate() calls
        avg_processing_time = self.total_inference_time / len(results) if results else 0.0

        return {
            "extraction_name": self.name,
            "total_items": len(results),
            "successful_items": len(successful),
            "failed_items": len(failed),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_input_tokens": round(avg_input_tokens, 2),
            "avg_output_tokens": round(avg_output_tokens, 2),
            "avg_processing_time": round(avg_processing_time, 4),
            "total_inference_time": round(self.total_inference_time, 4),
            "total_cost": round(total_cost, 6),
            "errors": [r.get("error") for r in failed if r.get("error")]
        }


    def format_summary(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a human-readable summary."""
        lines = []
        lines.append(_c(f"=== {metrics['extraction_name'].title()} Metrics ===", "bold"))

        # Basic stats
        lines.append(f"Total items: {metrics['total_items']}")
        lines.append(f"Successful: {metrics['successful_items']} ({metrics['success_rate']:.1%})")
        if metrics['failed_items'] > 0:
            lines.append(_c(f"Failed: {metrics['failed_items']}", "yellow"))

        # Timing (from LLM inference only)
        if metrics.get('total_inference_time', 0) > 0:
            lines.append(f"Total inference time: {metrics['total_inference_time']:.4f}s")
        if metrics.get('avg_processing_time', 0) > 0:
            lines.append(f"Avg processing time: {metrics['avg_processing_time']:.4f}s per item")

        # Token usage
        if metrics['total_tokens'] > 0:
            lines.append(_c("Token Usage:", "blue"))
            lines.append(f"  Input: {metrics['total_input_tokens']:,} (avg: {metrics['avg_input_tokens']:.1f})")
            lines.append(f"  Output: {metrics['total_output_tokens']:,} (avg: {metrics['avg_output_tokens']:.1f})")
            lines.append(f"  Total: {metrics['total_tokens']:,}")

        # Cost
        if metrics['total_cost'] > 0:
            lines.append(_c(f"Estimated cost: ${metrics['total_cost']:.6f}", "green"))

        return "\n".join(lines)


class SchemaMetricsCollector(BaseMetricsCollector):
    """Schema-based metrics collector that analyzes objects against a Pydantic schema."""

    def __init__(self, schema_class: Type[BaseModel], name: str = "schema_analysis"):
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required for SchemaMetricsCollector")
        if not issubclass(schema_class, BaseModel):
            raise ValueError("schema_class must be a Pydantic BaseModel")

        super().__init__(name)
        self.schema_class = schema_class
        self.schema_name = schema_class.__name__

    def collect_metrics(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect comprehensive metrics from objects based on schema."""
        if not objects:
            return {
                "schema_name": self.schema_name,
                "total_objects": 0,
                "field_metrics": {}
            }

        # Get schema field information
        field_metrics = {}
        model_fields = self.schema_class.model_fields

        for field_name, field_info in model_fields.items():
            # Extract values for this field from all objects
            field_values = [obj.get(field_name) for obj in objects]

            # Analyze this field
            field_metrics[field_name] = self._analyze_field(
                field_name, field_info, field_values
            )

        return {
            "schema_name": self.schema_name,
            "total_objects": len(objects),
            "field_metrics": field_metrics,
            "extraction_name": self.name
        }

    def _analyze_field(self, field_name: str, field_info: FieldInfo, values: List[Any]) -> Dict[str, Any]:
        """Analyze a single field across all objects."""
        # Get the field type
        field_type = field_info.annotation

        # Handle Optional types (Union[T, None])
        origin = get_origin(field_type)
        args = get_args(field_type)

        is_optional = False
        actual_type = field_type

        if origin is Union:
            # Check if it's Optional (Union with None)
            if len(args) == 2 and type(None) in args:
                is_optional = True
                actual_type = args[0] if args[1] is type(None) else args[1]

        # Count None values
        none_count = sum(1 for v in values if v is None)
        non_none_values = [v for v in values if v is not None]

        base_metrics = {
            "type": self._get_type_name(actual_type),
            "total_count": len(values),
            "non_none_count": len(non_none_values),
            "none_count": none_count,
            "is_optional": is_optional
        }

        # If all values are None, return basic metrics
        if not non_none_values:
            return base_metrics

        # Analyze based on the actual type
        type_metrics = self._get_type_analyzer(actual_type)(non_none_values, actual_type)

        return {**base_metrics, **type_metrics}

    def _get_type_name(self, type_hint: Type) -> str:
        """Get a readable name for a type."""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is list:
            if args:
                return f"list[{self._get_type_name(args[0])}]"
            return "list"
        elif origin is dict:
            if len(args) == 2:
                return f"dict[{self._get_type_name(args[0])}, {self._get_type_name(args[1])}]"
            return "dict"
        elif hasattr(type_hint, '__name__'):
            return type_hint.__name__
        else:
            return str(type_hint)

    def _get_type_analyzer(self, field_type: Type) -> Callable:
        """Return appropriate analyzer function for the field type."""
        origin = get_origin(field_type)

        # Handle generic types
        if origin is list:
            return self._analyze_list
        elif origin is dict:
            return self._analyze_dict

        # Handle basic types
        if field_type in (int, float) or (hasattr(field_type, '__name__') and field_type.__name__ in ('int', 'float')):
            return self._analyze_numeric
        elif field_type is bool or (hasattr(field_type, '__name__') and field_type.__name__ == 'bool'):
            return self._analyze_boolean
        elif field_type is str or (hasattr(field_type, '__name__') and field_type.__name__ == 'str'):
            return self._analyze_string

        # Handle Pydantic models (nested objects)
        if PYDANTIC_AVAILABLE and hasattr(field_type, '__bases__') and BaseModel in field_type.__bases__:
            return self._analyze_nested_object

        # Default to generic analysis
        return self._analyze_generic

    def _analyze_numeric(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Analyze numeric fields (int, float)."""
        if not values:
            return {}

        # Filter out non-numeric values and convert to float for calculations
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                continue

        if not numeric_values:
            return {"invalid_values": len(values)}

        try:
            return {
                "mean": round(statistics.mean(numeric_values), 4),
                "median": round(statistics.median(numeric_values), 4),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "std_dev": round(statistics.stdev(numeric_values), 4) if len(numeric_values) > 1 else 0,
                "range": max(numeric_values) - min(numeric_values),
                "percentile_25": round(statistics.quantiles(numeric_values, n=4)[0], 4) if len(numeric_values) >= 4 else None,
                "percentile_75": round(statistics.quantiles(numeric_values, n=4)[2], 4) if len(numeric_values) >= 4 else None,
                "valid_numeric_count": len(numeric_values),
                "invalid_values": len(values) - len(numeric_values)
            }
        except statistics.StatisticsError:
            return {"valid_numeric_count": len(numeric_values), "invalid_values": len(values) - len(numeric_values)}

    def _analyze_boolean(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Analyze boolean fields."""
        if not values:
            return {}

        true_count = sum(1 for v in values if v is True)
        false_count = sum(1 for v in values if v is False)
        invalid_count = len(values) - true_count - false_count

        total_valid = true_count + false_count

        return {
            "true_count": true_count,
            "false_count": false_count,
            "true_percentage": round(true_count / total_valid * 100, 2) if total_valid > 0 else 0,
            "false_percentage": round(false_count / total_valid * 100, 2) if total_valid > 0 else 0,
            "valid_boolean_count": total_valid,
            "invalid_values": invalid_count
        }

    def _analyze_string(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Analyze string fields."""
        if not values:
            return {}

        # Filter to actual strings
        string_values = [str(v) for v in values if isinstance(v, str)]
        invalid_count = len(values) - len(string_values)

        if not string_values:
            return {"invalid_values": len(values)}

        lengths = [len(s) for s in string_values]
        empty_count = sum(1 for s in string_values if len(s) == 0)

        return {
            "avg_length": round(statistics.mean(lengths), 2) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "median_length": round(statistics.median(lengths), 2) if lengths else 0,
            "empty_string_count": empty_count,
            "valid_string_count": len(string_values),
            "invalid_values": invalid_count,
            "total_characters": sum(lengths)
        }

    def _analyze_list(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Analyze list fields with deep nested analysis."""
        if not values:
            return {}

        # Filter to actual lists
        list_values = [v for v in values if isinstance(v, list)]
        invalid_count = len(values) - len(list_values)

        if not list_values:
            return {"invalid_values": len(values)}

        lengths = [len(lst) for lst in list_values]
        empty_count = sum(1 for lst in list_values if len(lst) == 0)

        metrics = {
            "avg_length": round(statistics.mean(lengths), 2) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "median_length": round(statistics.median(lengths), 2) if lengths else 0,
            "empty_list_count": empty_count,
            "valid_list_count": len(list_values),
            "invalid_values": invalid_count,
            "total_items": sum(lengths)
        }

        # Analyze list items with deep nested analysis
        if list_values:
            # Flatten all items from all lists
            all_items = []
            for lst in list_values:
                all_items.extend(lst)

            if all_items:
                # Try to get item type from type annotation
                args = get_args(field_type)
                item_type = args[0] if args else None

                # Perform deep analysis on items
                item_metrics = self._analyze_collection_items(all_items, item_type)
                if item_metrics:
                    metrics["item_metrics"] = item_metrics

        return metrics

    def _analyze_dict(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Analyze dictionary fields with deep nested analysis."""
        if not values:
            return {}

        # Filter to actual dictionaries
        dict_values = [v for v in values if isinstance(v, dict)]
        invalid_count = len(values) - len(dict_values)

        if not dict_values:
            return {"invalid_values": len(values)}

        key_counts = [len(d.keys()) for d in dict_values]
        empty_count = sum(1 for d in dict_values if len(d) == 0)

        # Collect all keys and values for deep analysis
        all_keys = set()
        key_value_map = {}  # key -> list of values for that key

        for d in dict_values:
            all_keys.update(d.keys())
            for key, value in d.items():
                if key not in key_value_map:
                    key_value_map[key] = []
                key_value_map[key].append(value)

        metrics = {
            "avg_key_count": round(statistics.mean(key_counts), 2) if key_counts else 0,
            "min_key_count": min(key_counts) if key_counts else 0,
            "max_key_count": max(key_counts) if key_counts else 0,
            "median_key_count": round(statistics.median(key_counts), 2) if key_counts else 0,
            "empty_dict_count": empty_count,
            "valid_dict_count": len(dict_values),
            "invalid_values": invalid_count,
            "unique_keys": list(all_keys),
            "unique_key_count": len(all_keys)
        }

        # Analyze values for each key with deep nested analysis
        if key_value_map:
            key_metrics = {}
            for key, key_values in key_value_map.items():
                key_analysis = self._analyze_collection_items(key_values, None)
                if key_analysis:
                    key_metrics[str(key)] = key_analysis

            if key_metrics:
                metrics["key_value_metrics"] = key_metrics

        return metrics

    def _analyze_nested_object(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Analyze nested Pydantic model fields."""
        if not values:
            return {}

        # Filter to actual dictionaries (nested objects are typically dicts)
        dict_values = [v for v in values if isinstance(v, dict)]
        invalid_count = len(values) - len(dict_values)

        if not dict_values:
            return {"invalid_values": len(values)}

        # Create a nested metrics collector for the sub-schema
        nested_collector = SchemaMetricsCollector(field_type, f"nested_{field_type.__name__}")
        nested_metrics = nested_collector.collect_metrics(dict_values)

        return {
            "nested_schema": field_type.__name__,
            "valid_object_count": len(dict_values),
            "invalid_values": invalid_count,
            "nested_metrics": nested_metrics["field_metrics"]
        }

    def _analyze_collection_items(self, items: List[Any], expected_type: Optional[Type] = None) -> Dict[str, Any]:
        """Perform deep analysis on collection items, handling nested structures."""
        if not items:
            return {}

        # Group items by their actual Python type
        type_groups = {}
        for item in items:
            item_type = type(item).__name__
            if item_type not in type_groups:
                type_groups[item_type] = []
            type_groups[item_type].append(item)

        analysis = {
            "total_items": len(items),
            "type_distribution": {k: len(v) for k, v in type_groups.items()},
            "unique_types": len(type_groups)
        }

        # Analyze each type group with appropriate metrics
        type_analyses = {}
        for type_name, type_items in type_groups.items():
            if type_name == 'str':
                type_analyses[type_name] = self._analyze_string(type_items, str)
            elif type_name in ('int', 'float'):
                type_analyses[type_name] = self._analyze_numeric(type_items, float)
            elif type_name == 'bool':
                type_analyses[type_name] = self._analyze_boolean(type_items, bool)
            elif type_name == 'list':
                # Recursively analyze nested lists
                nested_analysis = {}
                all_nested_items = []
                for nested_list in type_items:
                    if isinstance(nested_list, list):
                        all_nested_items.extend(nested_list)

                if all_nested_items:
                    nested_analysis = self._analyze_collection_items(all_nested_items)

                type_analyses[type_name] = {
                    "count": len(type_items),
                    "avg_length": round(statistics.mean([len(lst) for lst in type_items if isinstance(lst, list)]), 2) if type_items else 0,
                    "nested_items": nested_analysis
                }
            elif type_name == 'dict':
                # Recursively analyze nested dictionaries
                all_dict_values = []
                all_dict_keys = []
                for nested_dict in type_items:
                    if isinstance(nested_dict, dict):
                        all_dict_values.extend(nested_dict.values())
                        all_dict_keys.extend(nested_dict.keys())

                nested_analysis = {}
                if all_dict_values:
                    nested_analysis["values"] = self._analyze_collection_items(all_dict_values)
                if all_dict_keys:
                    nested_analysis["keys"] = self._analyze_collection_items(all_dict_keys)

                type_analyses[type_name] = {
                    "count": len(type_items),
                    "avg_key_count": round(statistics.mean([len(d) for d in type_items if isinstance(d, dict)]), 2) if type_items else 0,
                    "nested_analysis": nested_analysis
                }
            elif expected_type and PYDANTIC_AVAILABLE and hasattr(expected_type, '__bases__') and BaseModel in expected_type.__bases__:
                # Handle nested Pydantic models
                dict_items = [item for item in type_items if isinstance(item, dict)]
                if dict_items:
                    nested_collector = SchemaMetricsCollector(expected_type, f"nested_{expected_type.__name__}")
                    nested_metrics = nested_collector.collect_metrics(dict_items)
                    type_analyses[type_name] = {
                        "count": len(type_items),
                        "nested_schema_metrics": nested_metrics["field_metrics"]
                    }
            else:
                # Generic analysis for unknown types
                type_analyses[type_name] = {
                    "count": len(type_items),
                    "sample_values": type_items[:5] if len(type_items) <= 5 else type_items[:3] + ["..."]
                }

        if type_analyses:
            analysis["type_analyses"] = type_analyses

        return analysis

    def _analyze_generic(self, values: List[Any], field_type: Type) -> Dict[str, Any]:
        """Generic analysis for unknown types with deep collection analysis."""
        if not values:
            return {}

        # Use the deep collection analysis for generic types
        return self._analyze_collection_items(values, field_type)

    def format_summary(self, metrics: Dict[str, Any]) -> str:
        """Format schema metrics into a human-readable summary."""
        lines = []
        lines.append(_c(f"=== {metrics['schema_name']} Schema Analysis ===", "bold"))

        lines.append(f"Total objects analyzed: {metrics['total_objects']}")

        lines.append("")
        lines.append(_c("Field Metrics:", "blue"))

        for field_name, field_metrics in metrics.get('field_metrics', {}).items():
            lines.append(f"\n{_c(field_name, 'cyan')} ({field_metrics['type']}):")
            lines.append(f"  Total: {field_metrics['total_count']}, Non-null: {field_metrics['non_none_count']}, Null: {field_metrics['none_count']}")

            # Add type-specific metrics
            if 'mean' in field_metrics:  # Numeric
                lines.append(f"  Mean: {field_metrics['mean']}, Median: {field_metrics['median']}")
                lines.append(f"  Range: {field_metrics['min']} - {field_metrics['max']}")
            elif 'true_count' in field_metrics:  # Boolean
                lines.append(f"  True: {field_metrics['true_count']} ({field_metrics['true_percentage']}%)")
                lines.append(f"  False: {field_metrics['false_count']} ({field_metrics['false_percentage']}%)")
            elif 'avg_length' in field_metrics:  # String or List
                lines.append(f"  Avg length: {field_metrics['avg_length']}")
                lines.append(f"  Length range: {field_metrics['min_length']} - {field_metrics['max_length']}")
            elif 'nested_schema' in field_metrics:  # Nested object
                lines.append(f"  Nested schema: {field_metrics['nested_schema']}")
                lines.append(f"  Valid objects: {field_metrics['valid_object_count']}")

        return "\n".join(lines)


# Utility functions for creating metrics collectors
def create_metrics_collector(
    collector_type: str = "standard",
    name: str = "extraction",
    cost_per_input_token: float = 0.0,
    cost_per_output_token: float = 0.0,
    schema_class: Optional[Type[BaseModel]] = None,
    **kwargs
) -> BaseMetricsCollector:
    """Factory function to create metrics collectors."""
    if collector_type == "standard":
        return LLMMetricsCollector(
            name=name,
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token
        )
    elif collector_type == "schema":
        if schema_class is None:
            raise ValueError("schema_class is required for schema collector type")
        return SchemaMetricsCollector(
            schema_class=schema_class,
            name=name
        )
    else:
        raise ValueError(f"Unknown collector type: {collector_type}")


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    """Save metrics to JSON file."""
    from extraction.io import write_json
    write_json(output_path, metrics)


def load_metrics_json(input_path: str) -> Optional[Dict[str, Any]]:
    """Load metrics from JSON file."""
    from extraction.io import read_json
    return read_json(input_path)
