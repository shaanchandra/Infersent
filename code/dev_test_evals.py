def model_eval(epoch, eval_type, FLAGS):
    
    correct = 0
    
    if eval_type == 'dev':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = dev['s1'] if eval_type == 'dev' else test['s1']
    s2 = dev['s2'] if eval_type == 'dev' else test['s2']
    label = dev['label'] if eval_type == 'dev' else test['label']
    
    n_iter = int(np.ceil(len(dev['label'])/FLAGS.bsize))-1 if eval_type == 'dev' else int(np.ceil(len(test['label'])/FLAGS.bsize)) 
    
    for i in range(n_iter):
        
        # prepare batch
        start = i*FLAGS.bsize
        stop = start + FLAGS.bsize

        s1_batch, s1_lens = get_batch_from_idx(s1[start:stop], embeddings, config)
        s2_batch, s2_lens = get_batch_from_idx(s2[start:stop], embeddings, config)
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        label_batch = Variable(torch.LongTensor(label[start: stop])).to(device)

        # model forward
        if FLAGS.model_name == 'base':

            u = torch.sum(s1_batch, 0).to(device)
            v = torch.sum(s2_batch, 0).to(device)
            features = torch.cat((u, v, torch.abs(u- v), u*v), 1)
            output = classif(features).to(device)
            PATH = os.path.join(FLAGS.checkpoint_path,FLAGS.model_name)
            
        else:

            output = model(((s1_batch, s1_lens), (s2_batch, s2_lens)))
            PATH = os.path.join(FLAGS.checkpoint_path,FLAGS.model_name)
            
        
        pred = output.data.max(1)[1]
        correct += pred.long().eq(label_batch.data.long()).cpu().sum()
            
    print('saving model at epoch {0}'.format(epoch))

    
    
    # save model
    
    if FLAGS.model_name == 'base':
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        torch.save(classif.state_dict(), os.path.join(PATH, FLAGS.outputmodelname))
    else:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        
        torch.save(model.state_dict(), os.path.join(PATH, FLAGS.outputmodelname))

    
    eval_acc = round(100 * correct.item() / len(s1), 2)
    print("Validation accuracy = ", eval_acc)
                
    return eval_acc