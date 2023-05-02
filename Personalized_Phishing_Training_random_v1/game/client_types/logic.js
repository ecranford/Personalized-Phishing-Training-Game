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

//get packages to write data.csv file at end of phase 3
var stringify = require('csv-stringify/sync');
var fs = require('fs');

module.exports = function(treatmentName, settings, stager, setup, gameRoom) {

    let node = gameRoom.node;
    let channel = gameRoom.channel;
    let memory = node.game.memory;
    //Load database of emails
    var emails = require("../../emails.json");
    //console.log(emails);

    //var to store and check mturk ids
    var mturk_ids;
    
    // Setting the SOLO rule: game steps each time node.done() is called,
    // ignoring the state of other clients.
    stager.setDefaultStepRule(ngc.stepRules.SOLO);

    // Disabling step syncing for other clients: the logic does not
    // push step updates to other clients when it changes step.
    stager.setDefaultProperty('syncStepping', false);

    // Must implement the stages here.

    stager.setOnInit(function() {
        // Initialize the client.

        //load mturk ids
        mturk_ids = require("../../mturk-ids.json");
        //console.log("unparsed mturk-ids are: "+mturk_ids);

        // Will automatically save every entry in the database
        // to file memory.json (format ndjson).
        memory.stream();

        //create data object for each player in registry to use across levels
        this.pl.each(function(p) {
        channel.registry.updateClient(p.id, { data: [] }); //create object to store data
        channel.registry.updateClient(p.id, { total_timeout: 0 }); //create object to track total timeouts, and set value to 0
        channel.registry.updateClient(p.id, { consec_timeout: 0 }); //create object to track consectutive timeouts, and set value to 0
        channel.registry.updateClient(p.id, { ExitCode: J.uniqueKey({}, J.randomString(6, 'aA1')) }); //creates unique ExitCode 6 char string of random num/letters
        channel.registry.updateClient(p.id, { group: gameRoom.name.substring(4)}); //creates goup id from room number
        channel.registry.updateClient(p.id, { startTime: Date.now()}); //create object to save player start time
        channel.registry.updateClient(p.id, { repeat: false}); //create object to save player repeated boolean
        });

        node.on('in.say.PLAYER_UPDATE', function(msg) {
            if (msg.text === 'stage') {
                setTimeout(function() {
                    node.game.gotoStep(msg.data.stage);
                });
            }
        });

        // Last instruction in the init function.
        // Game on clients must be started manually
        // (because syncStepping is disabled).
        setTimeout(function() {
            node.remoteCommand('start', node.game.pl.first().id);
        });

        node.on.pdisconnect(function(player) {
            //console.log(player);
            if (player.disconnected) {
                //don't allow player to reconnect if kicked
                player.allowReconnect = false;

                if (player.repeat) {
                    node.game.stop();
                } else {
                    //player data will save when bot finishes, but...
                    //also need to compute current bonus whenever a player disconnects...but probably will just pay them base payment?
                    //Saves bonus file, and notifies players.
                    console.log("Saving data for "+player.id);
                    gameRoom.computeBonus({say: false, amt: true, addDisconnected: true, append: true, clients: [player.id]});
                    // Save times of all stages in case need to figure out how much base pay to pay them
                    memory.select('player', '=', player.id).save(player.id+'_times.csv', {
                        header: [
                            'session', 'player', 'stage', 'step', 'round', 'timestamp',
                            'time', 'timeup'
                        ]
                    });
                    // Save data to csv
                    memory.select('recordType', '=', 'decision').and('player', '=', player.id).save(player.id+'_data.csv', {
                        header: [
                            'session', 'group', 'player', 'WorkerId', 'mturkid','type',
                            'stage', 'step', 'round', 'time', 'timestamp','phase','trial',
                            'email_id', 'email_type','class_val','classification',
                            'class_time','confidence','conf_time','accuracy'
                        ]
                    });

                    let totalTime = Date.now()-player.startTime;
                    let bonus = [{
                        id: player.id,
                        type: player.clientType,
                        workerid: player.WorkerId,
                        hitid: player.HITId,
                        assignmentid: player.AssignmentId,
                        access: 'NA',
                        exit: player.ExitCode,
                        totaltime: totalTime,
                        approve: 1,
                        reject: 0,
                        basepay: node.game.settings.BASE_PAY,
                        bonus: player.winRaw*node.game.settings.EXCHANGE_RATE,
                        totalpay: (totalTime/60000)*0.10, //time in minutes * $0.10 per minute, no bonus
                        disconnected: player.disconnected == null ? 0 : player.disconnected,
                        disconnectStage: player.disconnectedStage.stage+'.'+player.disconnectedStage.step+'.'+player.disconnectedStage.round
                    }];
                    // Save bonus info to main bonus.csv
                    if (fs.existsSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv')) {
                        let output = stringify.stringify(bonus,{header: false});
                        try {
                            fs.appendFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv', output, 'utf8');
                            console.log('Bonus data saved for player '+player.id);
                        } catch (err) {
                            console.log('Some error occured - file either not saved or corrupted file saved for player '+player.id);
                        };
                    } else {
                        let output = stringify.stringify(bonus,{header: true});
                        try {
                            fs.writeFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv', output, 'utf8');
                            console.log('Bonus data saved for player '+player.id);
                        } catch (err) {
                            console.log('Some error occured - file either not saved or corrupted file saved for player '+player.id);
                        };
                    };
                    //console.log(node.game.pl.id.getAllKeys());
                    node.game.stop();
                };
            };
        });

        //console.log("player list:\n"+this.pl);
        //console.log("channel registry clients:\n"+channel.registry.clients.player.getAllKeys());

    });

    stager.extendStep('terms and conditions', {
        cb: function() {
            node.on.done(function(msg) {
                let data = msg.data;

                channel.registry.updateClient(data.player, { WorkerId: data.WorkerId });
                channel.registry.updateClient(data.player, { mturkid: data.forms.mturkid.value });
                channel.registry.updateClient(data.player, { AssignmentId: data.AssignmentId });
                channel.registry.updateClient(data.player, { HITId: data.HITId });

                //two ways to access client object
                //console.log(this.pl.get(data.player));
                //console.log(channel.registry.getClient(data.player));
                let player = this.pl.get(data.player);

                //if WorkerId or mturkid are already in database, then redirect player, but don't want to save any data
                //else save WorkerId and mturkid
                let hasWorkerId = false;
                let hasMturkId = false;
                for (let i = 0; i < mturk_ids.length; i++) {
                    if(mturk_ids[i].workerid == data.WorkerId){hasWorkerId = true; break;}
                };

                for (let i = 0; i < mturk_ids.length; i++) {
                    if(mturk_ids[i].mturkid == data.forms.mturkid.value){hasMturkId = true; break;}
                };

                if (hasWorkerId || hasMturkId) {
                    player.repeat = true;
                    node.redirect('repeated.htm', data.player);
                    //alternative method to disconnect player
                    //node.remoteAlert('Records indicate you have already participated.\nYou will be disconnected.', data.player);
                    //node.disconnectClient(player);
                } else {
                    mturk_ids.push({workerid: data.WorkerId, mturkid: data.forms.mturkid.value});
                    //console.log("mturk ids are now: "+mturk_ids);
                    let myjson = JSON.stringify(mturk_ids);
                    //console.log("myjson is: "+myjson);
                    try {
                        fs.writeFileSync('./games_available/Personalized_Phishing_Training_random_v1/mturk-ids.json', myjson, 'utf8');
                        console.log('mturkid saved for player '+data.player);
                    } catch (err) {
                        console.log('Some error occured - mturkid not saved or corrupted file saved for player '+data.player);
                    };
                }
            });
        }
    });

    stager.extendStep('practice trial', {
        cb: function() {
            this.pl.each(function(p) {
                //choose which email to present on each trial of practice
                var email = emails.filter(el => {return el['id'] === "ex"+node.game.getRound();});
                node.say("email", p.id, email);
            });
        }
    });

    stager.extendStep('start game', {
        exit: function() {
            console.log("----------Beginning Phase 1----------");
        }
    })

    //choose which email to present on each trial of pre-test (Phase 1)
    stager.extendStep('phase 1', {
        init: function() {
            let trial = node.game.getRound();
            let phase = node.game.getCurrentStepObj();
            console.log("------------------------");
            console.log(phase.id[0].toUpperCase() + phase.id.substring(1));
            console.log("Trial "+trial);
            //send phase, trial, and email data to each participant
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //change this selection for CogModel and RMAB versions so that it requests and gets info from model.
                let seen_emails_list = player.data.filter(item => item.email_id);
                //console.log("Seen emails list from registry: "+player.data.filter(item => item.email_id));
                let seen_emails = [];
                for (let i = 0; i < seen_emails_list.length; i++) {
                    //console.log(scores[i].accuracy);
                    seen_emails.push(seen_emails_list[i].email_id);
                };
                console.log(p.id+" Seen emails has duplicates: "+((new Set(seen_emails)).size !== seen_emails.length));
                let seen_email_types_list = player.data.filter(item => item.phase === "phase 1");
                let seen_email_types = [];
                for (let i = 0; i < seen_email_types_list.length; i++) {
                    //console.log(scores[i].accuracy);
                    seen_email_types.push(seen_email_types_list[i].email_type);
                };
                //console.log(p.id+" Seen email types: "+seen_email_types);
                let num_phishing = seen_email_types.filter(x => x=="PHISHING").length;
                console.log(p.id+" Number phishing: "+num_phishing);
                let num_ham = seen_email_types.filter(x => x=="HAM").length;
                console.log(p.id+" Number ham: "+num_ham);
                let selected_email_type;
                let email_num;
                //for pre- and post-test phases, send random 10 phishing and 10 ham
                if (num_phishing <= 9) {
                    if (num_ham <= 9) {
                        if (Math.floor(Math.random() * 2) == 0) {
                            selected_email_type = "PHISHING";
                            console.log("Both <= 9, sending phishing to "+p.id);
                        } else {
                            selected_email_type = "HAM";
                            console.log("Both <= 9, sending ham to "+p.id);
                        };
                    } else {
                        selected_email_type = "PHISHING";
                        console.log("10 ham, sending phishing to "+p.id);
                    };
                  } else {
                    selected_email_type = "HAM";
                    console.log("10 phishing, sending ham to "+p.id);
                };
                if (selected_email_type == "PHISHING") {
                    do {
                        email_num = Math.floor(Math.random() * 188) + 1;
                    }
                    while (seen_emails.includes(''+email_num));
                    
                } else {
                    do {
                        email_num = Math.floor(Math.random() * 177) + 189;
                    }
                    while (seen_emails.includes(''+email_num));
                }; 
                console.log("Fetching email number "+email_num+" for "+p.id);

                node.say("phasedata", p.id, [phase.id[0].toUpperCase() + phase.id.substring(1), trial]);
                var email = emails.filter(el => {return el['id'] === email_num.toString();});
                if (email_num.toString() !== email[0].id) {
                    console.log("Error: wrong email retrieved from database for "+p.id);
                }
                //console.log("Verifying Email ID "+email[0].id+" for "+p.id);
                //console.log("Email sender: "+email[0].sender);
                //console.log("Email subject: "+email[0].subject);
                //console.log("Email body: "+email[0].body);
                p.email_id = email[0].id;
                p.email_type = email[0].type;
                node.say("email", p.id, email);
            });
            
        },
        cb: function () {
            node.on.done(function(msg) {
                let data = msg.data;

                let player = channel.registry.getClient(data.player);

                let acc;
                let choice = data.forms.classification.choice;
                
                let email_id = this.pl.get(data.player).email_id;
                let email_type = this.pl.get(data.player).email_type;
                let classification;
                let confidence_val = data.forms.confidence.value;
                //algorithm for 50-100 scale
                //((0.01*data.forms.confidence.value)*50)+50;

                switch (choice) {
                    case 0:
                        if (email_type == 'PHISHING') {acc = 1.0;} else {acc = 0.0;};
                        classification = 'PHISHING';
                        break;
                    case 1:
                        if (email_type == 'HAM') {acc = 1.0;} else {acc = 0.0;};
                        classification = 'HAM';
                        break;
                    default:
                        acc = 0.0;
                        classification = 'HAM';
                };
                console.log(data.player+" Classification: "+choice+" "+classification);
                console.log(data.player+" Confidence: "+confidence_val);
                console.log(data.player+" Accuracy: "+acc);

                // Update earnings counts, so that it can be saved
                // with GameRoom.computeBonus.
                gameRoom.updateWin(data.player, acc);

                //save data to memory
                memory.add({recordType: "decision",
                    player: data.player,
                    WorkerId: player.WorkerId,
                    mturkid: player.mturkid,
                    type: player.clientType,
                    session: data.session,
                    group: player.group,
                    stage: data.stage,
                    time: data.time,
                    timestamp: data.timestamp,
                    timeup: +data.timeup,
                    phase: data.stepId,
                    trial: data.stage.round,
                    email_id: email_id,
                    email_type: email_type,
                    class_val: choice,
                    classification: classification,
                    class_time: data.forms.classification.time,
                    confidence: confidence_val,
                    conf_time: data.forms.confidence.time,
                    accuracy: acc
                });
                
                //save data to registry
                player.data.push({
                    player: data.player,
                    WorkerId: player.WorkerId,
                    mturkid: player.mturkid,
                    type: player.clientType,
                    session: data.session,
                    group: player.group,
                    stage: data.stage.stage,
                    step: data.stage.step,
                    round: data.stage.round,
                    time: data.time,
                    timestamp: data.timestamp,
                    timeup: +data.timeup,
                    phase: data.stepId,
                    trial: data.stage.round,
                    email_id: email_id,
                    email_type: email_type,
                    class_val: choice,
                    classification: classification,
                    class_time: data.forms.classification.time,
                    confidence: confidence_val,
                    conf_time: data.forms.confidence.time,
                    accuracy: acc   
                });

                //track timeouts in player object
                if (typeof player.total_timeout === 'number') {
                    if (data.timeup && typeof choice != 'number') {
                        player.total_timeout = player.total_timeout + 1;
                        player.consec_timeout = player.consec_timeout + 1;
                    } else {
                        player.consec_timeout = 0;
                    };
                } else {
                    if (typeof choice === 'number') {
                        player.total_timeout = 0;
                        player.consec_timeout = 0;
                    } else {
                        player.total_timeout = +data.timeup;
                        player.consec_timeout = +data.timeup; 
                    }
                };
                console.log(data.player+" Total timeouts: "+player.total_timeout);
                console.log(data.player+" Consecutive timeouts: "+player.consec_timeout);
                //player.id is player id
                //player.sid is player sid
                //console.log(node.nodename); //room name

                //disconnect if 3 consecutive timeouts or 5 total timeouts
                if (player.total_timeout >= 5 || player.consec_timeout >= 3) {
                    //disconnect player here
                    //console.log(node.game.pl.id.getAllKeys());
                    console.log(data.player+" was removed");
                    //Redirect player to disconnected page?
                    node.redirect('disconnected.htm', player.id);
                    //node.redirect('disconnected.htm'+'?code='+player.ExitCode, player.id);
                    
                    //alternative method to disconnecte a player
                    //node.remoteAlert('You have been disconnected due to inactivity.\nPlease return to the HIT and enter this code to receive your partial payment: ', player.id);
                    //node.disconnectClient(player);
                    
                    //console.log(node.game.pl.id.getAllKeys());
                    node.game.pause();
                    //node.game.stop();
                    //node.game.resume();
                    //node.game.gameover();
                };
                

            });
        }
    });

    // Phase 1 feedback
    stager.extendStep('phase 1 feedback', {
        init: function() {
            let prev_phase = node.game.getPreviousStep();
            let phase = node.game.getStageId(prev_phase);
            let phase_name = phase[0].toUpperCase() + phase.substring(1);
            //send phase and score data to each participant
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //phase data
                node.say("phasedata", p.id, phase_name);
                //phase score data
                let phase_scores = player.data.filter(item => item.phase === "phase 1");
                let phase_score = 0;
                for (let i = 0; i < phase_scores.length; i++) {
                    //console.log(scores[i].accuracy);
                    phase_score += phase_scores[i].accuracy;
                };
                //total score data
                let total_score = 0;
                for (let i = 0; i < player.data.length; i++) {
                    //console.log(scores[i].accuracy);
                    total_score += player.data[i].accuracy;
                }
                console.log(p.id+" "+phase_name+" Score: "+phase_score);
                console.log(p.id+" Total Score: "+total_score);
                node.say("scores", p.id, [phase_score,total_score]);
            });
        },
        cb: function() {
            // Save times of all stages.
            memory.done.save('times.csv', {
                header: [
                    'session', 'player', 'stage', 'step', 'round', 'timestamp',
                    'time', 'timeup'
                ]
            });
            // Save data to csv
            memory.select('recordType', '=', 'decision').save('data.csv', {
                header: [
                    'session', 'group', 'player', 'WorkerId', 'mturkid','type',
                    'stage', 'step', 'round', 'time', 'timestamp','phase','trial',
                    'email_id', 'email_type','class_val','classification',
                    'class_time','confidence','conf_time','accuracy'
                ]
            });

            node.on.data('level_done', function(msg) {
                // currentRoom is optional, avoid lookup.
                let currentRoom; // let currentRoom = gameRoom.name; 
                let levelName = 'Phase2';
                // Move client to the next level.
                // (async so that it finishes all current step operations).
                setTimeout(function() {
                    console.log('moving client to next level: ', msg.from);
                    channel.moveClientToGameLevel(msg.from, levelName, currentRoom);
                }, 100);
            });
        }
    });

};
