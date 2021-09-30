var mqtt = require("mqtt");
var client = mqtt.connect("mqtt://192.168.1.3:5112", {
  clientId: "Device_1",
  clean: false,
});
var topic = "/lightTime";

client.on("message", (topic, message) => {
  message = message.toString();
  console.log("Subscriber received ::" + message);
});

client.on("connect", () => {
  client.subscribe(topic);
});
