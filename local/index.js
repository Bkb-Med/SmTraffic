var admin = require("firebase-admin");

var serviceAccount = require("./serviceAccount.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://smarttraffic-3116d-default-rtdb.firebaseio.com",
});

var db = admin.database();
var ref = db.ref().child("smarttraffic-3116d-default-rtdb");

ref
  .limitToLast(1)
  .once("value")
  .then(function (snapshot) {
    snapshot.forEach(function (childSnapshot) {
      console.log(childSnapshot.val());
    });
  });
