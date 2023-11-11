import json
nose_etalon = 180
sup_st_etalon = 90
sup_fin_etalon = 180
def estim(file_save:str, nose_array,support_array,time,fps):
    try:
        with open(file_save,'r') as json_file:
            json_decoded = json.load(json_file)
    except FileNotFoundError:
        print('file not found')
        json_decoded={}

    st_n,fin_n=time
    if not fps:
        key=f'frame {st_n} : {fin_n}'
    else:
        key = f' time {round(st_n/fps,3)}s   : {round(fin_n/fps,3)}s'
    nose=nose_array[0]
    sup_st=support_array[0]
    sup_fin=support_array[-1]
    nose_estim=round(max (0,nose_etalon-nose),3)
    sup_st_estim=round(max(0,sup_st-sup_st_etalon),3)
    sup_fin_estim=round(max(0,sup_fin_etalon-sup_fin),3)
    total=round(nose_estim+sup_st_estim+sup_fin_estim,3)
    # estim={'nose_estim': nose_estim,'sup_st_estim': sup_st_estim,
    #        'sup_fin_estim': sup_fin_estim,'total':total}
    # value={'nose':nose,'sup_st':sup_st,'sup_fin':sup_fin}
    # all_dict={'estimations':estim,'value':value}
    estim = {'nose':nose,'sup_st':sup_st,'sup_fin':sup_fin,'nose_estim': nose_estim, 'sup_st_estim': sup_st_estim,
                    'sup_fin_estim': sup_fin_estim,'total':total}
    json_decoded[key]=estim
    # json_decoded[key]=all_dict
    with open(file_save, 'w') as json_file:
        json.dump(json_decoded, json_file)
