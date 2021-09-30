import ssl
import datetime
ssl.match_hostname = lambda cert, hostname: True

class MQTTClient():
   
   def __init__(self,client):
      #super(MQTTClient, self).__init__(cname,**kwargs)
      self.client = client
      self.broker_host = "192.168.1.4"
      self.port = 8883
      self.topic_data = "Trafficlight/duration"
      self.topic_state = "Trafficlight/State"
      self.message_state = False
      

   def on_message(self, client, userdata, message):
       self.message_state = True
       print("|=================================================================|")
       print("| "+str(datetime.datetime.now())+" | received message #", str(message.payload.decode("utf-8"))+"|")


   def on_publish(self,client, userdata, result):  # create function for callback
        print("| "+str(datetime.datetime.now())+" | data published "+"|")
        print("|=================================================================|")
        pass

   def connect(self):
       self.client.tls_set('ca.crt')
       self.client.connect(self.broker_host, self.port)
       print("| connected to... "+str(self.broker_host)+":"+str(self.port))
       self.client.loop_start()
       print("| subscribing...")
       self.client.subscribe(self.topic_data)
       self.client.on_message = self.on_message
       self.client.on_publish = self.on_publish

   def stop_connection(self):
       self.client.disconnect() #disconnect
       self.client.loop_stop()  #stop loop
       
   def publish(self):

       if self.message_state == True:
            print("| publishing... ")
            self.client.publish(self.topic_state, "message received")
            self.message_state = False



