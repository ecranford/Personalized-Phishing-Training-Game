/**
 * # Logic type implementation of the game stages
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * http://www.nodegame.org
 * ---
 */

"use strict";

const ngc = require('nodegame-client');
const J = ngc.JSUS;

module.exports = function(treatmentName, settings, stager, setup, gameRoom) {

    let node = gameRoom.node;
    let channel = gameRoom.channel;
    let memory = node.game.memory;
    var emails = require("C:/home/nodegame-v7.1.0/games_available/Personalized_Phishing_Training_random_v1/emails.json");
    //console.log(emails);

    // Must implement the stages here.

    stager.setOnInit(function() {
        // Initialize the client.
        // Will automatically save every entry in the database
        // to file memory.json (format ndjson).
        memory.stream();
        //Load database of emails

    });
/*
    //choose which email to present on each trial of practice
    stager.extendStep('practice trials', {
        cb: function() {
        this.pl.each(function(p) {
                var email = emails.filter(el => {return el['id'] === "ex"+node.game.getRound();});
                node.say("email", p.id, email);
            });
        }
    });
    */
    //choose which email to present on each trial of pre-test (Phase 1)
    stager.extendStep('phase 1', {
        init: function() {
            let trial = node.game.getRound();
            let phase = node.game.getCurrentStepObj();
            console.log(phase.id[0].toUpperCase() + phase.id.substring(1));
            console.log(trial);
            //send phase, trial, and email data to each participant
            this.pl.each(function(p) {
                
                let email_num = Math.floor(Math.random() * 4) + 1;

                node.say("phasedata", p.id, [phase.id[0].toUpperCase() + phase.id.substring(1), trial]);
                var email = emails.filter(el => {return el['id'] === "ex"+email_num;});
                console.log(email);
                p.email_id = email[0].id;
                p.email_type = email[0].type;
                node.say("email", p.id, email);

                console.log(p);
            });
            
        },
        cb: function () {
            node.once.done(function(msg) {
                let data = msg.data;
                console.log(data);
                let acc;
                let choice = data.forms.classification.choice;
                console.log(choice);
                let email_id = this.pl.get(data.player).email_id;
                let email_type = this.pl.get(data.player).email_type;
                let classification;

                switch (choice) {
                    case 0:
                        if (email_type == 'PHISHING') {acc = 1;} else {acc = 0;};
                        classification = 'PHISHING';
                        break;
                    case 1:
                        if (email_type == 'HAM') {acc = 1;} else {acc = 0;};
                        classification = 'HAM';
                        break;
                    default:
                        acc = 0;
                };
                console.log(acc);

                //save data to file
                node.game.memory.add({recordType: "decision",
                    player: data.player,
                    session: data.session,
                    stage: data.stage,
                    time: data.time,
                    timestamp: data.timestamp,
                    phase: data.stepId,
                    trial: data.stage.round,
                    email_id: email_id,
                    email_type: email_type,
                    class_val: choice,
                    classification: classification,
                    class_time: data.forms.classification.time,
                    confidence: data.forms.confidence.intialValue + data.forms.confidence.value,
                    conf_time: data.forms.confidence.time,
                    accuracy: acc
                    })
            });
        }
    });

    ///TEMPLATE STAGES BELOW HERE --- NEED TO MODIFY STILL
    stager.extendStep('training', {
        matcher: {
            roles: [ 'DICTATOR', 'OBSERVER' ],
            match: 'round_robin',
            cycle: 'mirror_invert',
            // sayPartner: false
            // skipBye: false,

        },
        cb: function() {
            node.once.done(function(msg) {
                let data = msg.data;
                let offer = data.offer;

                // Send the decision to the other player.
                node.say('decision', data.partner, offer);

                // Update earnings counts, so that it can be saved
                // with GameRoom.computeBonus.
                gameRoom.updateWin(msg.from, settings.COINS - offer);
                gameRoom.updateWin(data.partner, offer);

            });
        }
    });

    stager.extendStep('end', {
        init: function() {

            // Feedback.
            memory.view('feedback').stream({
                header: [ 'time', 'timestamp', 'player', 'feedback' ],
                format: 'csv'
            });

            // Email.
            memory.view('email').stream({
                header: [ 'timestamp', 'player', 'email' ],
                format: 'csv'
            });

        },
        cb: function() {

            // Saves bonus file, and notifies players.
            gameRoom.computeBonus();

            // Dump all memory.
            // memory.save('memory_all.json');

            // Save times of all stages.
            memory.done.save('times.csv', {
                header: [
                    'session', 'player', 'stage', 'step', 'round',
                    'time', 'timeup'
                ]
            });
        }
    });

    stager.setOnGameOver(function() {
        // Something to do.
    });
};
