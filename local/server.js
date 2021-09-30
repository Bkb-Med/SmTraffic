//mqtt broker
var admin = require("firebase-admin");
var mosca = require("mosca");

var serviceAccount = require("./serviceAccount.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://smarttraffic-3116d-default-rtdb.firebaseio.com",
});

var db = admin.database();
var ref = db.ref().child("smarttraffic-3116d-default-rtdb/trafficData");

var SECURE_KEY = __dirname + "/pem/server.key";
var SECURE_CERT = __dirname + "/pem/server.crt";

var settings = {
  port: 8883,
  logger: {
    name: "Smart traffic",
    level: 40,
  },
  secure: {
    keyPath: SECURE_KEY,
    certPath: SECURE_CERT,
  },
};
var server = new mosca.Server(settings);

server.on("ready", setup);

server.on("published", (packet) => {
  let date_ob = new Date();
  let date = ("0" + date_ob.getDate()).slice(-2);
  let month = ("0" + (date_ob.getMonth() + 1)).slice(-2);
  let year = date_ob.getFullYear();
  let hours = date_ob.getHours();
  let minutes = date_ob.getMinutes();
  let seconds = date_ob.getSeconds();
  let fulldate =
    year +
    "-" +
    month +
    "-" +
    date +
    " " +
    hours +
    ":" +
    minutes +
    ":" +
    seconds;

  console.log(
    "|=================================================================|"
  );
  console.log(
    "|" + fulldate + "| Topic published :: " + packet.topic.toString()
  );
  console.log(
    "|" + fulldate + "| message on server :: " + packet.payload.toString()
  );
  console.log(
    "|=================================================================|"
  );
});

server.on("clientConnected", function (client) {
  console.log("| Client connected |", client.id);
});

server.on("clientDisconnected", function (client) {
  console.log("| Client Disconnected |", client.id);
});

var messageFromserver = {
  topic: "Trafficlight/duration",
  payload: 0, // or a Buffer
  qos: 1, // 0, 1, or 2
  retain: false, // or true
};

function setup() {
  console.log("=====server is up and running===___ENSIAS@2020__miola====");
  ref.on("child_added", function (snapshot) {
    var changedPost = snapshot.val();
    console.log(
      "Valeur provient du cloud database " +
        changedPost.valeur +
        "<" +
        changedPost.State +
        ">"
    );
    messageFromserver.payload =
      changedPost.valeur.toString() + "<s>" + changedPost.State.toString();
    server.publish(messageFromserver);
  });
}
