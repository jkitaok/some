# Schema Metrics

Automatic analysis of structured data against Pydantic schemas with comprehensive statistics and validation.

## Features

- **Type-Aware Analysis**: Automatic metrics for numeric, boolean, string, list, and nested fields
- **Data Validation**: Identifies invalid values and schema violations
- **None Handling**: Proper tracking of optional fields and missing values
- **Nested Objects**: Recursive analysis of complex schemas
- **Performance Metrics**: Timing and efficiency statistics

## Quick Start

```python
from pydantic import BaseModel
from typing import List, Optional
from extraction.metrics import SchemaMetricsCollector

# Define schema
class Product(BaseModel):
    name: str
    price: float
    features: List[str]
    description: Optional[str] = None

# Analyze data
collector = SchemaMetricsCollector(Product, "analysis")
metrics = collector.collect_metrics(your_data)
print(collector.format_summary(metrics))
```

## Advanced Usage

### Factory Function
```python
from extraction.metrics import create_metrics_collector

collector = create_metrics_collector(
    collector_type="schema",
    schema_class=Product
)
```

### Nested Schemas
```python
class Order(BaseModel):
    customer: Customer  # Nested object
    products: List[Product]  # List of nested objects
    total: float

# Automatically analyzes all nested schemas
collector = SchemaMetricsCollector(Order, "order_analysis")
```

## Metrics Output

### Numeric Fields
- Mean, median, min, max, standard deviation
- Percentiles (25th, 75th)
- Valid/invalid value counts

### Boolean Fields
- True/false counts and percentages
- Null value tracking

### String Fields
- Length statistics (avg, min, max, median)
- Empty string detection
- Character counts

### List Fields
- Length statistics for lists
- Item-level analysis for typed lists
- Empty list tracking

### Nested Objects
- Recursive analysis of nested schemas
- Field-by-field breakdown
- Validation tracking

## Data Validation

Automatically tracks invalid values that don't match schema types:

```python
# Invalid data example
invalid_data = [{"price": "not_a_number"}]  # Should be float
metrics = collector.collect_metrics(invalid_data)

# Check validation results
for field, stats in metrics["field_metrics"].items():
    if stats.get("invalid_values", 0) > 0:
        print(f"{field}: {stats['invalid_values']} invalid values")
```

## Integration

```python
# Save results
from extraction.metrics import save_metrics_json
save_metrics_json(metrics, "results.json")

# Timing support
collector.start_timing()
metrics = collector.collect_metrics(data)
collector.end_timing()
```

## Testing

Run tests: `python -m unittest tests.metrics_test.TestSchemaMetricsCollector -v`
