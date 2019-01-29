import numpy as np
from flask import Flask,abort,jsonify, request
import pickle

My_Log_reg = pickle.load(open("log_reg.pkl","rb"))

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def make_predict():
	data = request.get_json(force=True)
	#convert our json to numpy array
	predict_request = [data['loan_amnt'],data['funded_amnt'],data['funded_amnt_inv'],data['term'],data['int_rate'],data['installment'],data['sub_grade'],data['emp_title'],data['emp_length'],data['home_ownership'],data['annual_inc'],data['verification_status'],data['issue_d'],data['loan_status'],data['purpose'],data['addr_state'],data['dti'],data['delinq_2yrs'],data['earliest_cr_line'],data['inq_last_6mths'],data['open_acc'],data['pub_rec'],data['revol_bal'],data['revol_util'],data['total_acc'],data['initial_list_status'],data['out_prncp'],data['out_prncp_inv'],data['total_pymnt'],data['total_pymnt_inv'],data['total_rec_prncp'],data['total_rec_int'],data['total_rec_late_fee'],data['recoveries'],data['collection_recovery_fee'],data['last_pymnt_d'],data['last_pymnt_amnt'],data['last_credit_pull_d'],data['collections_12_mths_ex_med'],data['policy_code'],data['application_type'],data['acc_now_delinq']]
	predict_request = np.array(predict_request)
	#np array into the model
	y_hat = My_Log_reg.predict(predict_request)
	output = [y_hat[0]]
	return jsonify(result=output)

if __name__== '__main__':
	app.run(port = 9000,debug = True)