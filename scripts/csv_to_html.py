#!/usr/bin/env python3
"""
Script to convert CSV data to a nicely formatted HTML table.
"""

import pandas as pd
import sys
import os
from datetime import datetime

def create_html_table(csv_file_path, output_file_path=None):
    """Convert CSV file to HTML table with nice styling."""
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # If no output path specified, create one based on input file
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_dir = os.path.dirname(csv_file_path)
        output_file_path = os.path.join(output_dir, f"{base_name}_table.html")
    
    # Round numeric columns to reasonable precision for display
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if col not in ['timestep', 'env_id']:  # Keep these as integers
            df[col] = df[col].round(4)
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RSL-RL Play Session Data - {os.path.basename(csv_file_path)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            background-color: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .table-container {{
            overflow-x: auto;
            padding: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0;
            font-size: 0.9em;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 12px 8px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e7f3ff;
        }}
        
        .numeric {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        
        .timestamp {{
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            color: #666;
        }}
        
        /* Color coding for different data types */
        .action-col {{
            background-color: rgba(102, 126, 234, 0.1);
        }}
        
        .obs-col {{
            background-color: rgba(118, 75, 162, 0.1);
        }}
        
        .meta-col {{
            background-color: rgba(40, 167, 69, 0.1);
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #dee2e6;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .stats {{
                flex-direction: column;
                gap: 15px;
            }}
            
            th, td {{
                padding: 8px 4px;
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RSL-RL Play Session Data</h1>
            <p>File: {os.path.basename(csv_file_path)}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">Total Timesteps</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{df['env_id'].nunique()}</div>
                <div class="stat-label">Environments</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(df.columns)}</div>
                <div class="stat-label">Data Columns</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{df['timestep'].max()}</div>
                <div class="stat-label">Max Timestep</div>
            </div>
        </div>
        
        <div class="table-container">
"""
    
    # Generate the table HTML with proper column styling
    html_content += "            <table>\n"
    html_content += "                <thead>\n                    <tr>\n"
    
    # Add headers with appropriate styling classes
    for col in df.columns:
        if col.startswith('action_'):
            class_name = 'action-col'
        elif col.startswith('obs_'):
            class_name = 'obs-col'
        else:
            class_name = 'meta-col'
        html_content += f'                        <th class="{class_name}">{col}</th>\n'
    
    html_content += "                    </tr>\n                </thead>\n"
    html_content += "                <tbody>\n"
    
    # Add data rows
    for _, row in df.iterrows():
        html_content += "                    <tr>\n"
        for col in df.columns:
            value = row[col]
            
            # Apply appropriate styling based on column type
            if col in ['timestep', 'env_id']:
                html_content += f'                        <td class="numeric meta-col">{value}</td>\n'
            elif col == 'timestamp':
                html_content += f'                        <td class="timestamp meta-col">{value}</td>\n'
            elif col.startswith('action_'):
                html_content += f'                        <td class="numeric action-col">{value}</td>\n'
            elif col.startswith('obs_'):
                html_content += f'                        <td class="numeric obs-col">{value}</td>\n'
            else:
                html_content += f'                        <td>{value}</td>\n'
        html_content += "                    </tr>\n"
    
    html_content += """                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>RSL-RL Play Session Data Visualization</p>
            <p>Actions are color-coded in blue, Observations in purple, Metadata in green</p>
        </div>
    </div>
</body>
</html>"""
    
    # Write the HTML file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML table created successfully: {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_html.py <csv_file_path> [output_html_path]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        sys.exit(1)
    
    result = create_html_table(csv_file, output_file)
    if result:
        print(f"\nOpen the following file in your browser to view the table:")
        print(f"file://{os.path.abspath(result)}")