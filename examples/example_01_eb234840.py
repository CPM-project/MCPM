from MCPM.multipletpf import MultipleTpf
from MCPM.campaigngridradec2pix import CampaignGridRaDec2Pix
from MCPM import utils
from MCPM.prfdata import PrfData
from MCPM.prfforcampaign import PrfForCampaign


if __name__ == "__main__":
    # We want to extract the light curve of OGLE-BLG-ECL-234840, which is
    # an eclipsing binary with a single eclipse in subcampaign 91. I know 
    # the star coordinates and that it is in channel 31, but you can find 
    # the channel this way:
    # >>> import K2fov
    # >>> K2fov.fields.getKeplerFov(9)pickAChannel(ra, dec)
    channel = 31
    campaign = 91
    ra = 269.929125
    dec = -28.410833
    
    tpfs = MultipleTpf()
    tpfs.campaign = campaign
    tpfs.channel = channel
    
    out = tpfs.get_predictor_matrix(ra=ra, dec=dec)
    (predictor_matrix, predictor_matrix_mask) = out
    print(predictor_matrix.shape)
    print(predictor_matrix_mask.shape, sum(predictor_matrix_mask))
    
    grids = CampaignGridRaDec2Pix(campaign=campaign, channel=channel)
    (mean_x, mean_y) = grids.mean_position(ra, dec)
    print("Mean target position: {:.2f} {:.2f}\n".format(mean_x, mean_y))

    half_size = 2
    pixels = utils.pixel_list_center(mean_x, mean_y, half_size)
    print(pixels)
    
    prf_template = PrfData(channel=channel)
    # Third, the highest level structure - something that combines grids and 
    # PRF data:
    prf_for_campaign = PrfForCampaign(campaign=campaign, grids=grids, 
                                    prf_data=prf_template)
                                    
    (prfs, mask_prfs) = prf_for_campaign.apply_grids_and_prf(ra, dec, pixels)  
    
    print(tpfs.get_epic_id_for_radec(ra, dec))