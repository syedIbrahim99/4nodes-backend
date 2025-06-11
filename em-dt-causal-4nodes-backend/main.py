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
    df = pd.read_csv("surface_4.csv")
    head_data = df.iloc[:200]
    surface_matrix = head_data.transpose()
    return surface_matrix


re_cal_result = None
    
async def re_calculation(original_B_matrix, surface_matrix, source_matrix, coupled_ref_matrix, websocket):
    n_co_opto = coupled_ref_matrix
    n_su_simo = surface_matrix
    k_fin = 100
    coupled_converge_to = 0
    k_co = []
    changed_nodes_history = []
    
    for k in range(1, k_fin+1):
        n_su_simn = np.dot(original_B_matrix, np.add(n_su_simo, source_matrix))
        n_co_optn = np.dot(original_B_matrix, n_su_simn)
        coupled_con_tn = np.linalg.norm(np.subtract(n_co_optn, n_co_opto))
        Cocd = np.abs(np.subtract(coupled_con_tn, coupled_converge_to))
        
        # Append the current values of k and Cocd to k_co
        k_co.append((k, Cocd))
        print("k_co", len(k_co), "Cocd", Cocd)
        if Cocd < k_co[0][1] / 100:
            break
        # print(k)
        
        changed_nodes = np.where(n_co_opto != n_co_optn)[0]
        changed_nodes_indices = original_B_matrix.index[changed_nodes]
        unique_changed_nodes = set(changed_nodes_indices)
        changed_nodes_strings = [str(node) for node in unique_changed_nodes] 
        # print("changed_nodes_strings", type(changed_nodes_strings))
        changed_nodes_history.append(changed_nodes_strings)

        # Update variables for next iteration
        n_co_opto = n_co_optn
        n_su_simo = n_su_simn
        coupled_converge_to = coupled_con_tn
    print("changed_nodes_history", changed_nodes_history)
    kco_array = np.array(k_co)
    print("kco_array", kco_array)

    formatted_convergencce = [{"x": int(item[0]), "y": item[1]} for item in kco_array]
    print("formatted_convergence", formatted_convergencce)
    return {
        "matrix": pd.DataFrame(n_co_opto, index=original_B_matrix.columns, columns=surface_matrix.columns),
        "blink": changed_nodes_history,
        "covergence_value": formatted_convergencce
        }
        


@app.websocket("/initial-matrix")
async def initial_matrix(websocket: WebSocket):
    global re_cal_result
    print("Accepting connection")
    await websocket.accept()
    print("Accepted")
    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            print(json_data)
            transposed_B_matrix = convert_dict_into_matrix(json_data)
            original_B_matrix = transposed_B_matrix.transpose()
            surface_matrix = prepare_surface_matrix()
            identity_matrix = np.eye(4)
            source_matrix = np.dot(np.subtract(identity_matrix, original_B_matrix), surface_matrix)
            coupled_ref_matrix = np.subtract(surface_matrix, source_matrix)
            
            # print("causal_matrix", causal_matrix)
            # print("surface_matrix", surface_matrix)
            # print("source_matrix", source_matrix.shape)
            # print("coupled_ref_matrix", coupled_ref_matrix)
            re_cal_result = await re_calculation(original_B_matrix, surface_matrix, source_matrix, coupled_ref_matrix, websocket)
            # print("re_cal_result", re_cal_result["blink"])
           
            if re_cal_result is not None:
                for i in range(re_cal_result["matrix"].shape[0]):
                    try:
                        result_dict = {
                            "node": re_cal_result["matrix"].index[i], 
                            "data": re_cal_result["matrix"].iloc[i].to_dict(),
                            "convergence": re_cal_result['covergence_value']
                            }
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



@app.websocket("/changed-matrix")
async def get_changed_matrix(websocket: WebSocket):
    global re_cal_result
    print("Accepting connection")
    await websocket.accept()
    print("Accepted")
    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            print(json_data)
            transposed_B_matrix = convert_dict_into_matrix(json_data)
            original_B_matrix = transposed_B_matrix.transpose()
            surface_matrix = prepare_surface_matrix()
            identity_matrix = np.eye(4)
            source_matrix = np.dot(np.subtract(identity_matrix, original_B_matrix), surface_matrix)
            coupled_ref_matrix = np.subtract(surface_matrix, source_matrix)
            
            # print("causal_matrix", causal_matrix)
            # print("surface_matrix", surface_matrix)
            # print("source_matrix", source_matrix.shape)
            # print("coupled_ref_matrix", coupled_ref_matrix)
            re_cal_result = await re_calculation(original_B_matrix, surface_matrix, source_matrix, coupled_ref_matrix, websocket)
            # print("re_cal_result", re_cal_result["blink"])
           
            if re_cal_result is not None:
                for i in range(re_cal_result["matrix"].shape[0]):
                    try:
                        result_dict = {
                            "node": re_cal_result["matrix"].index[i], 
                            "data": re_cal_result["matrix"].iloc[i].to_dict(), 
                            "node_blink":re_cal_result["blink"],
                            "convergence": re_cal_result['covergence_value']
                            }
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
        