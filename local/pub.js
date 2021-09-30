var mqtt = require("mqtt");
const fs = require("fs");

var options = {
  clientId: "LOCAL_DATA",
  clean: true,
  useSSL: true,
  rejectUnauthorized: false,
  //keyPath: KEY,
  //certPath: SECURE_CERT,
  ca: [fs.readFileSync(__dirname + "/pem/ca.crt")],
};

var client = mqtt.connect("mqtts://192.168.1.3:8883", options);

var topic = "Trafficlight/duration";

var num = 0;
client.on("connect", () => {
  console.log("connected flag  " + client.connected);
  setInterval(() => {
    var message = Math.floor(Math.random() * 6).toString();
    client.publish(topic, message, {
      qos: 2,
      retain: true,
    });
    console.log("message sent!", message + "[" + num++ + "]");
  }, 5000);
});

client.on("error", function (error) {
  console.log("Can't connect" + error);
  process.exit(1);
});
