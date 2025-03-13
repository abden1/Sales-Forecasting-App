from setuptools import setup, find_packages

setup(
    name="sales-forecasting",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==2.0.1',
        'pandas==1.5.3',
        'numpy==1.24.3',
        'scikit-learn==1.2.2',
        'plotly==5.13.0',
        'statsmodels==0.14.0',
        'flask-cors==4.0.0',
        'python-dotenv==1.0.0',
        'waitress==2.1.2'
    ],
    entry_points={
        'console_scripts': [
            'sales-forecast=app.app:main',
        ],
    },
) 