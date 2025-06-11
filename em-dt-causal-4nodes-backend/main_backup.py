from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import pandas as pd
import asyncio
import uvicorn
from dotenv import load_dotenv
import os
from starlette.websockets import WebSocketState
import logging
import numpy as np
import matplotlib.pyplot as plt
    


app = FastAPI()
load_dotenv(".env")
port = os.getenv("port")
host = os.getenv("host")
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def convert_dict_into_matrix(json_data):
    nodes = json_data['nodes']
    edge_labels = json_data['edge_labels']
    print(nodes, edge_labels)
    # Initialize a DataFrame with zeros
    matrix_df = pd.DataFrame(0, index=nodes, columns=nodes, dtype=float)

    # Fill in the DataFrame based on edge labels
    for edge, value in edge_labels.items():
        node1, node2 = edge.split(', ')
        matrix_df.at[node1, node2] = float(value)

    return matrix_df

def prepare_surface_matrix():
    df = pd.read_csv("surface.csv")
    head_data = df.iloc[:200]
    surface_matrix = head_data.transpose()
    return surface_matrix

# i=0
# def nodes_changed(n_co_opt_prev_iter, n_co_opto):
#     global i
#     i=i+1
#     print("n_co_opto", i)
#     diff = np.subtract(n_co_opt_prev_iter, n_co_opto)
#     with open('diff.txt', 'a') as file:
#     # Write each element of the list to the file
#         for item in diff:
#             file.write(str(item) + '\n')
#     return diff


re_cal_result = None
    
async def re_calculation(causal_matrix, surface_matrix, source_matrix, coupled_ref_matrix, websocket):
    n_co_opto = coupled_ref_matrix
    n_su_simo = surface_matrix
    k_fin = 100
    coupled_converge_to = 0
    # k_co = np.empty((0,))
    k_co = []
    changed_nodes_history = []
    
    # print("n_co_opto_1", n_co_opto)
    # with open('n_co_opto_1.txt', 'a') as file:
    #     # Write each element of the list to the file
    #     for item in n_co_opto.values:
    #         file.write(str(item) + '\n')
  
    for k in range(1, k_fin+1):

        n_su_simn = np.dot(causal_matrix, np.add(n_su_simo, source_matrix))
        n_co_optn = np.dot(causal_matrix, n_su_simn)
        coupled_con_tn = np.linalg.norm(np.subtract(n_co_optn, n_co_opto))
        Cocd = np.abs(np.subtract(coupled_con_tn, coupled_converge_to))
        
        # Append the current values of k and Cocd to k_co
        # print("k, Cocd", k, Cocd)
        # k_co = np.append(k_co, [k, Cocd])
        k_co.append((k, Cocd))
        print("k_co", k_co)
        if Cocd < k_co[0][1] / 100:
            break
        # print(k)
        
        changed_nodes = np.where(n_co_opto != n_co_optn)[0]
        changed_nodes_indices = causal_matrix.index[changed_nodes]
        unique_changed_nodes = set(changed_nodes_indices)
        changed_nodes_strings = [str(node) for node in unique_changed_nodes] 
        # print("changed_nodes_strings", type(changed_nodes_strings))
        changed_nodes_history.append(changed_nodes_strings)

        # Update variables for next iteration
        n_co_opto = n_co_optn
        n_su_simo = n_su_simn
        coupled_converge_to = coupled_con_tn
        
        # with open('n_co_opto_2.txt', 'a') as file:
        #     # Write each element of the list to the file
        #     for item in n_co_opto:
        #         file.write(str(item) + '\n')
    print("changed_nodes_history", changed_nodes_history)
    kco_array = np.array(k_co)
    print("kco_array", kco_array.shape)
    plt.figure()
    plt.plot(kco_array[:-1,0], kco_array[:-1,1])
    plt.show()
    flattened_list = [node for sublist in changed_nodes_history for node in sublist]
    return {"matrix": pd.DataFrame(n_co_opto, index=causal_matrix.columns, columns=surface_matrix.columns), "blink": flattened_list}
        


@app.websocket("/causal-matrix")
async def websocket_endpoint(websocket: WebSocket):
    global re_cal_result
    print("Accepting connection")
    await websocket.accept()
    print("Accepted")
    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            print(json_data)
            causal_matrix = convert_dict_into_matrix(json_data)
            surface_matrix = prepare_surface_matrix()
            identity_matrix = np.eye(12)
            source_matrix = np.dot(np.subtract(identity_matrix, causal_matrix), surface_matrix)
            coupled_ref_matrix = np.subtract(surface_matrix, source_matrix)
            
            # print("causal_matrix", causal_matrix)
            # print("surface_matrix", surface_matrix)
            # print("source_matrix", source_matrix.shape)
            # print("coupled_ref_matrix", coupled_ref_matrix)
            re_cal_result = await re_calculation(causal_matrix, surface_matrix, source_matrix, coupled_ref_matrix, websocket)
            # print("re_cal_result", re_cal_result["blink"])
           
            if re_cal_result is not None:
                for i in range(re_cal_result["matrix"].shape[0]):
                    try:
                        result_dict = {"node": re_cal_result["matrix"].index[i], "data": re_cal_result["matrix"].iloc[i].to_dict(), "node_blink":re_cal_result["blink"]}
                        # print("result_dict", result_dict)
                        await websocket.send_json(result_dict)
                    except WebSocketDisconnect:
                        print("Client disconnected")
                    except Exception as e:
                        error_message = f"Error sending timeseries data: {str(e)}"
                        print(error_message)
                        raise ValueError(error_message)
            else:
                print("No re_cal_result matrix available yet.")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        error_message = f"Error processing JSON data: {str(e)}"
        print(error_message)
        raise ValueError(error_message)

     
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=int(port))   
        