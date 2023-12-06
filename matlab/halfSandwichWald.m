function covB = halfSandwichWald(pinvDesignMtx, residual, groupIds, numGroupIDs)
    [numCovariates, ~] = size(pinvDesignMtx);
    [numObs, numFcEdges] = size(residual);
    covB = zeros(numCovariates,numCovariates,numFcEdges);
    for grpId = 0:(numGroupIDs-1)
        subjThisGrp = groupIds == grpId;
        halfSandwich = pinvDesignMtx(:, subjThisGrp) * residual(subjThisGrp,:);
        for fcEdgeIdx = 1:numFcEdges
            covB(:,:,fcEdgeIdx) = ...
                covB(:,:,fcEdgeIdx) + ...
                halfSandwich(:,fcEdgeIdx) * halfSandwich(:,fcEdgeIdx)';
        end
    end     
end
