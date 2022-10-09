"""
Overall layout idea:

We need an FlowEvents object, which is a wrapped Pandas dataframe.
The main dataframe is the raw event information to index into, with a separate
metadata column that accounts for each individual datapoint in a lookup table.

There needs to be a merge object that adds information to the metadata lookup.

Gate objects take Transforms. Transforms store information about columns on their own,
and can be applied to FlowEvents objects.

This way, Gates can be applied easily, even over more complicated transforms like
RatioTransforms.

A gate constructor looks like:
BlahGate(x_transform, y_transform, *, any_other_constants_needed)

FlowEvents has helpers that apply gates to themselves. This would look like
fevents.filter(BlahGate(LinearTransform('FCS-A'), LogTransform('mRuby2-A'), some_extra_params)).label()
except most of the time you would create your gates as:

singlets_gate = PolygonGate(LinearTransform('FCS-A'), LinearTransform('SSC-A')).edit()
and use the other software :)

but then you do

fevents.filter(singlets_gate).label(quad_gate).describe()

The Gate objects need to have column names and transforms attached, in addition to a type


"""