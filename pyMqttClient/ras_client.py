
import paho.mqtt.client as paho
import time
from mqttclient import MQTTClient as mqc



def main():
    client1 = paho.Client("RAS_CLIENT001")
    mqttClient = mqc(client1)
    print("=========== connecting to broker...============= ")
    print("| ClientID : RAS_CLIENT001")
    mqttClient.connect()
    

    while True:
        mqttClient.publish()
        time.sleep(0.01)

    mqttClient.stop_connection()   
    

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

