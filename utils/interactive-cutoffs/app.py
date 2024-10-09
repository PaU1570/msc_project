from flask import Flask, request, jsonify, render_template, redirect
import pandas as pd
import plotly.express as px
import data_utils as du
import os
import sys

app = Flask(__name__)
data_path = None
files_list = None

@app.route('/callback', methods=['POST', 'GET'])
def cb():
    file_num = int(request.args.get('file_num'))
    VStartPos = float(request.args.get('VStartPos'))
    VEndPos = float(request.args.get('VEndPos'))
    VStartNeg = float(request.args.get('VStartNeg'))
    VEndNeg = float(request.args.get('VEndNeg'))
    return file(file_num, VStartPos, VEndPos, VStartNeg, VEndNeg)

@app.route('/file/<int:file_num>')
def file(file_num, VStartPos=0.1, VEndPos=4, VStartNeg=-0.1, VEndNeg=-4):
    if files_list is None or file_num > len(files_list) or file_num < 1:
        return redirect('/')
    selected_file = files_list[file_num-1]
    metadata, meas_params, meas_data = du.read_file(selected_file)
    df = du.get_df_1(meas_data, VStartPos=0.1, VEndPos=4, VStartNeg=-0.1, VEndNeg=-3)
    fig = px.scatter(df, x="Pulse Amplitude (V)", y="R_high (ohm)", color="Keep", color_discrete_sequence=['red', 'blue'])
    graphJSON1 = fig.to_json()

    df = du.get_df_2(meas_data, VStartPos=0.1, VEndPos=4, VStartNeg=-0.1, VEndNeg=-3)
    fig = px.scatter(df, x="Normalized Pulse Number", y="Normalized Conductance", color="Type", color_discrete_sequence=['red', 'blue'])
    graphJSON2 = fig.to_json()

    return render_template('cutoff_plots.html',
                           graphJSON1=graphJSON1,
                           graphJSON2=graphJSON2,
                           metadata=metadata,
                           meas_params=meas_params,
                           file=selected_file, 
                           file_num=file_num,
                           total_file_num=len(files_list),
                           VStartPos=VStartPos,
                           VEndPos=VEndPos,
                           VStartNeg=VStartNeg,
                           VEndNeg=VEndNeg
                           )
  
@app.route('/')
def index():
    global files_list
    global data_path

    if files_list is None:
        if len(sys.argv) < 1:
            return 'No data folder path provided'
        
        data_path = sys.argv[1]
        files_list = du.get_csv_files(data_path)
    return render_template('index.html', data_path=data_path, files_list=files_list, file_num=len(files_list))

if __name__ == '__main__':
    app.run(debug=True)