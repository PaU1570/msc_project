from flask import Flask, request, jsonify, render_template, redirect
import plotly.express as px
import data_utils as du
import sys
import csv

app = Flask(__name__)
data_path = None
files_list = None
selected_file = None
selected_file_num = None
metadata = None
meas_params = None
meas_data = None

output_file = None

keys = ['device_name', 'device_id', 'test_date', 'test_time', 'VStartPos', 'VEndPos', 'VStartNeg', 'VEndNeg']
cutoffs = dict() # key: file number; value: list in the following order: [device_name, device_id, test_date, test_time, VStartPos, VEndPos, VStartNeg, VEndNeg]

@app.route('/callback1', methods=['POST', 'GET'])
def cb1():
    VStartPos = float(request.args.get('VStartPos'))
    VEndPos = float(request.args.get('VEndPos'))
    VStartNeg = float(request.args.get('VStartNeg'))
    VEndNeg = float(request.args.get('VEndNeg'))

    graphJSON1 = graph1(meas_data, VStartPos, VEndPos, -VStartNeg, -VEndNeg)

    return graphJSON1

@app.route('/callback2', methods=['POST', 'GET'])
def cb2():
    VStartPos = float(request.args.get('VStartPos'))
    VEndPos = float(request.args.get('VEndPos'))
    VStartNeg = float(request.args.get('VStartNeg'))
    VEndNeg = float(request.args.get('VEndNeg'))

    graphJSON2 = graph2(meas_data, VStartPos, VEndPos, -VStartNeg, -VEndNeg)

    return graphJSON2

@app.route('/save', methods=['POST', 'GET'])
def save():
    VStartPos = float(request.args.get('VStartPos'))
    VEndPos = float(request.args.get('VEndPos'))
    VStartNeg = -float(request.args.get('VStartNeg'))
    VEndNeg = -float(request.args.get('VEndNeg'))

    cutoffs[selected_file_num] = [metadata['device_name'], metadata['device_id'], metadata['test_date'], metadata['test_time'], VStartPos, VEndPos, VStartNeg, VEndNeg]

    success = True
    try:
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(cutoffs.values())
    except Exception as e:
        print(e)
        success = False

    res = jsonify({"status": "success"}) if success else jsonify({"status": "failed"})
    return res

@app.route('/file/<int:file_num>')
def file(file_num, VStartPos=0.1, VEndPos=4, VStartNeg=0.1, VEndNeg=4):
    global selected_file
    global selected_file_num
    global metadata
    global meas_params
    global meas_data

    if files_list is None or file_num > len(files_list) or file_num < 1:
        return redirect('/')
    selected_file = files_list[file_num-1]
    selected_file_num = file_num
    metadata, meas_params, meas_data = du.read_file(selected_file)

    graphJSON1 = graph1(meas_data, VStartPos, VEndPos, -VStartNeg, -VEndNeg)
    graphJSON2 = graph2(meas_data, VStartPos, VEndPos, -VStartNeg, -VEndNeg)

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
    global output_file

    if files_list is None:       
        files_list = du.get_csv_files(data_path)
    return render_template('index.html', data_path=data_path, files_list=files_list, file_num=len(files_list))

def graph1(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg):
    df = du.get_df_1(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg)
    fig = px.scatter(df, x="Pulse Amplitude (V)", y="R_high (ohm)", color="Keep", color_discrete_sequence=['red', 'blue'])
    fig.add_vline(x=VStartPos, line_dash="dash", line_color="green", annotation_text="VStartPos")
    fig.add_vline(x=VEndPos, line_dash="dash", line_color="green", annotation_text="VEndPos")
    fig.add_vline(x=VStartNeg, line_dash="dash", line_color="orange", annotation_text="VStartNeg")
    fig.add_vline(x=VEndNeg, line_dash="dash", line_color="orange", annotation_text="VEndNeg")
    graphJSON1 = fig.to_json()
    return graphJSON1

def graph2(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg):
    df = du.get_df_2(meas_data, VStartPos, VEndPos, VStartNeg, VEndNeg)
    fig = px.scatter(df, x="Normalized Pulse Number", y="Normalized Conductance", color="Type", color_discrete_sequence=['red', 'blue'])
    graphJSON2 = fig.to_json()
    return graphJSON2


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python app.py <data_path> <output_file>")
        sys.exit(1)
    data_path = sys.argv[1]
    output_file = sys.argv[2]

    app.run(debug=True)