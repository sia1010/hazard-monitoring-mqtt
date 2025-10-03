import datetime
import os

import paho.mqtt.client as mqtt

broker = 'mqtt-dashboard.com'
port = 1883
topic_send = "hazard-monitoring/client"
topic_recv = "hazard-monitoring/server"
client_id = "central-hub"

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, message):
    client.publish(topic_send, message)

def subscribe(client):
    def on_message(client, userdata, msg):
        message = msg.payload.decode()
        record = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "," + message + "\n"
        f = open("data.csv", mode='a')
        f.write(record)
        f.close()

    client.subscribe(topic_recv)
    client.on_message = on_message

def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_start()
    os.system("py -m streamlit run webapp.py")
    while True:
        pass


if __name__ == '__main__':
    run()
