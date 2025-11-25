# IVI Water Trends - Interactive Visualization and Impact Assessment Tool

A Python-based system for analyzing seasonal surface water trends and assessing the impact of natural resource management (NRM) interventions using CoRE Stack data.

## 🌊 Features

- **Data Integration**: Fetch seasonal surface water data from CoRE Stack APIs
- **Impact Assessment**: Analyze the effectiveness of farm ponds, check dams, and other NRM interventions
- **Interactive Visualizations**: Create stunning charts and dashboards with Plotly
- **Comprehensive Reports**: Generate PDF reports and Short-friendly summaries
- **CLI Tools**: Batch processing and automation capabilities
- **Extensible Design**: Modular architecture for easy customization

## 📋 What's Included

### Core Modules
- **API Client** (`ivi_water/api_client.py`): CoRE Stack integration with caching and retry logic
- **Data Processor** (`ivi_water/data_processor.py`): Data cleaning, merging, and statistical analysis
- **Visualizer** (`ivi_water/visualizer.py`): Interactive charts and 3D visualizations
- **Export Utils** (`ivi_water/export_utils.py`): Report generation and data export
- **CLI Interface** (`ivi_water/cli.py`): Command-line tools for batch processing

### Sample Data & Notebooks
- **Sample Data**: Water trends and NRM impact datasets for testing
- **Getting Started**: Complete demonstration script
- **Documentation**: Comprehensive setup and usage guides

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ivi-water/ivi-water-trends.git
cd ivi-water-trends

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .   # Remember the dot after -e
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# CORE_API_KEY=your_api_key_here
# CORE_API_BASE_URL=https://api.corestack.org/v1
# DATA_DIR=./data
# OUTPUT_DIR=./outputs
```

### 3. Run Demo

```bash
# Run the getting started demo
python notebooks/getting_started.py

# Or use CLI commands
ivi-water get-spatial-units --unit-type village
ivi-water fetch-water-data --locations V001,V002 --start-year 2020 --end-year 2022
ivi-water visualize --data water_data.csv --chart-type seasonal
```

## 📊 Usage Examples

### Python API

```python
from ivi_water import CoREStackClient, DataProcessor, WaterTrendsVisualizer

# Initialize components
client = CoREStackClient()
processor = DataProcessor()
viz = WaterTrendsVisualizer()

# Fetch data
water_data = processor.load_water_data_from_api(
    client, ['V001', 'V002'], 2020, 2022
)

# Create visualizations
fig = viz.create_seasonal_stacked_area_chart(water_data)
fig.show()
```

### Command Line Interface

```bash
# Get spatial units
ivi-water get-spatial-units --unit-type village --state Maharashtra

# Fetch water data
ivi-water fetch-water-data --locations V001,V002 --start-year 2020 --end-year 2022

# Merge with NRM data
ivi-water merge-data --water-data water.csv --nrm-data nrm.csv

# Create visualizations
ivi-water visualize --data merged.csv --chart-type seasonal

# Generate reports
ivi-water generate-report --data merged.csv --report-type summary
```

## 📈 Visualizations

### Available Chart Types
- **Seasonal Stacked Area Charts**: Water availability trends over time
- **Intervention Comparison Plots**: Before/after analysis
- **Water Body Distribution**: Statistical histograms
- **Trend Heatmaps**: Spatial-temporal patterns
- **3D Visualizations**: Optional PyVista-based 3D water body rendering

### Dashboard Features
- Multi-location comparison
- Interactive filtering
- Export capabilities (HTML, PNG, SVG, PDF)
- Responsive design

## 📋 Reports

### Report Types
- **Summary Reports**: Overview with key insights
- **Detailed Reports**: Comprehensive analysis with charts
- **Short Summaries**: Mobile-friendly key takeaways

### Export Formats
- PDF reports with charts and statistics
- Excel workbooks with multiple sheets
- CSV data exports
- HTML interactive dashboards

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CORE_API_KEY` | CoRE Stack API key | Required |
| `CORE_API_BASE_URL` | API endpoint | `https://api.corestack.org/v1` |
| `DATA_DIR` | Data files directory | `./data` |
| `OUTPUT_DIR` | Output files directory | `./outputs` |
| `CACHE_TTL` | Cache duration (seconds) | `3600` |
| `DEFAULT_CHART_THEME` | Plotly theme | `plotly_white` |

### Data Requirements

#### Water Data Format
```csv
location_id,year,season,water_area_ha,water_body_count,data_quality
V001,2020,perennial,45.2,8,good
V001,2020,winter,32.1,6,good
```

#### NRM Impact Data Format
```csv
location_id,year,pond_presence,crop_yield_ton_per_ha,drought_sensitivity
V001,2020,1,2.8,low
V001,2021,1,3.1,low
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=ivi_water tests/

# Run specific test
pytest tests/test_api_client.py
```

## 📚 Documentation

- **Getting Started Guide**: `notebooks/getting_started.py`
- **API Documentation**: Inline docstrings in all modules
- **CLI Help**: `ivi-water --help`
- **Sample Data**: `data/sample/`

## 🏗️ Architecture

```
ivi_water/
├── api_client.py      # CoRE Stack API integration
├── data_processor.py  # Data cleaning and analysis
├── visualizer.py      # Visualization components
├── export_utils.py    # Report generation
└── cli.py            # Command-line interface

data/
├── sample/           # Sample datasets
├── raw/             # Raw data files
└── processed/       # Processed data

notebooks/            # Analysis notebooks
outputs/             # Generated reports and charts
tests/               # Unit tests
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Run linting
flake8 ivi_water/
black ivi_water/

# Run type checking
mypy ivi_water/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `docs/` folder
- **Issues**: Report bugs on GitHub Issues
  
## 🙏 Acknowledgments

- **CoRE Stack**: For providing the water data APIs
- **Plotly**: For amazing visualization capabilities
- **Pandas**: For powerful data manipulation
- **Open Source Community**: For inspiration and best practices

---

**Built with ❤️ for rural landscape planning and water resource management**
