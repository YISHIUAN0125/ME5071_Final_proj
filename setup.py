from setuptools import setup, find_packages

setup(
    name="ME5071_final_project",
    version="0.1.0",
    description="ME5071 final project",
    packages=find_packages(),
    python_requires='>=3.8',
    include_package_data=True,
    # entry_points={
    #     "console_scripts": [
    #         "damask-train=tools.train:main"
    #     ]
    # },
)
